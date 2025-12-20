#!/usr/bin/env python3
# Project Euler 765 — Trillionaire
#
# No external libraries are used.
#
# Key idea:
#   View the game under a "fair coin" measure (win/lose equally likely).
#   Under that measure the wealth process is a martingale for ANY betting strategy,
#   hence E_Q[X_1000] is fixed (= initial wealth).
#   Becoming a trillionaire requires terminal wealth >= M, which consumes at least
#   M units of wealth on each successful outcome-path.
#   Therefore, with initial wealth 1 you can afford at most floor(2^n / M) success paths.
#   To maximize the *real* success probability (biased coin with p=0.6), you should
#   spend this limited "success budget" on the most likely paths: those with the
#   largest number of wins.
#
# The optimal success probability can then be computed exactly using big integers.

from __future__ import annotations


def binom_coeffs(n: int) -> list[int]:
    "Return [C(n,0), C(n,1), ..., C(n,n)] as exact integers."
    c = [1] * (n + 1)
    for k in range(1, n + 1):
        # C(n,k) = C(n,k-1) * (n-k+1) / k
        c[k] = c[k - 1] * (n - k + 1) // k
    return c


def rounded_decimal(num: int, den: int, digits: int) -> str:
    """
    Round (num/den) to `digits` digits after the decimal point, using integer arithmetic.
    Returns a string like '0.1234567890'.
    """
    if den <= 0:
        raise ValueError("denominator must be positive")
    if num < 0:
        raise ValueError("numerator must be nonnegative")

    scale = 10**digits
    q, r = divmod(num * scale, den)
    # Round half up
    if r * 2 >= den:
        q += 1
    int_part, frac_part = divmod(q, scale)
    return f"{int_part}.{str(frac_part).zfill(digits)}"


def solve() -> None:
    n = 1000
    M = 10**12  # target wealth in grams of gold

    # Real coin probabilities: p = 3/5, q = 2/5.
    # We'll compute exact probabilities as integers over denominator 5^n.
    pq_den = 5

    # Total number of outcome paths for n coin tosses.
    total_paths = 1 << n

    # Under the fair-coin measure, each path has probability 1/2^n.
    # If we guarantee X_N >= M on a set S of paths, then E[X_N] >= M*|S|/2^n.
    # But E[X_N] is always 1 under the fair-coin measure, so:
    #   |S| <= floor(2^n / M)
    budget_paths = total_paths // M
    assert budget_paths > 0

    comb = binom_coeffs(n)

    # Sanity: sum_k C(n,k) == 2^n
    assert sum(comb) == total_paths

    # Suffix counts: suffix[k] = number of paths with >= k wins.
    suffix = [0] * (n + 2)
    for k in range(n, -1, -1):
        suffix[k] = suffix[k + 1] + comb[k]

    # Choose paths with the most wins first.
    # Find k0 such that:
    #   suffix[k0]  >  budget_paths >= suffix[k0+1]
    # Then we take ALL paths with wins > k0 (i.e., >= k0+1),
    # and 'rem' paths among those with exactly k0 wins.
    k0 = None
    for k in range(n, -1, -1):
        if suffix[k] > budget_paths >= suffix[k + 1]:
            k0 = k
            break
    assert k0 is not None

    rem = budget_paths - suffix[k0 + 1]
    assert 0 <= rem < comb[k0]

    # Precompute powers of 2 and 3 up to n (exact ints).
    pow2 = [1] * (n + 1)
    pow3 = [1] * (n + 1)
    for i in range(1, n + 1):
        pow2[i] = pow2[i - 1] * 2
        pow3[i] = pow3[i - 1] * 3

    den = pq_den**n  # 5^n

    # Full contribution from all paths with k >= k0+1 wins:
    num = 0
    for k in range(k0 + 1, n + 1):
        # There are C(n,k) paths; each has probability 3^k 2^(n-k) / 5^n.
        num += comb[k] * pow3[k] * pow2[n - k]

    # Partial contribution from 'rem' paths with exactly k0 wins:
    # each such path has probability 3^k0 2^(n-k0) / 5^n.
    if rem:
        num += rem * pow3[k0] * pow2[n - k0]

    # Output rounded to 10 digits after the decimal point.
    print(rounded_decimal(num, den, 10))


if __name__ == "__main__":
    solve()
