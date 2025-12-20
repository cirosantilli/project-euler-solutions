#!/usr/bin/env python3
"""
Project Euler 900 - DistribuNim II

No external libraries, single-threaded.

We use:
- A closed-form for t(n) based on a power-of-two divisibility rule.
- A short linear recurrence for S(N) (verified against exact small values),
  then iterate it up to N = 10^4 modulo 900497239.
"""

from __future__ import annotations


MOD = 900497239
TARGET_N = 10_000


def next_power_of_two_strictly_greater(x: int) -> int:
    """
    Return the smallest power of two strictly greater than x, for x >= 1.
    Using bit_length:
      - if x is a power of two, this returns 2*x
      - otherwise it returns the next power of two above x
    """
    return 1 << x.bit_length()


def t(n: int) -> int:
    """
    Let p be the smallest power of 2 strictly greater than n.
    Then the minimal k >= 0 that makes the special (n+1)-pile position losing is:
        t(n) = (-n^2 - (n mod 2)) mod p
    """
    p = next_power_of_two_strictly_greater(n)
    return (-n * n - (n & 1)) % p


def exact_S_up_to(Nmax: int) -> list[int]:
    """
    Compute exact S(k) = sum_{n=1}^{2^k} t(n) for k = 0..Nmax (S(0)=0),
    by direct summation using the closed form for t(n).
    This is only used for small Nmax to seed/verify the recurrence.
    """
    if Nmax < 0:
        raise ValueError("Nmax must be nonnegative")

    S = [0] * (Nmax + 1)
    if Nmax == 0:
        return S

    total = 0
    next_cut = 2  # 2^1
    k = 1
    limit = 1 << Nmax
    for n in range(1, limit + 1):
        total += t(n)
        if n == next_cut:
            S[k] = total
            k += 1
            next_cut <<= 1
            if k > Nmax:
                break
    return S


def S_mod_large(N: int, mod: int = MOD) -> int:
    """
    Compute S(N) modulo mod for large N using a verified order-5 linear recurrence:
        S(n)=7S(n-1)-6S(n-2)-48S(n-3)+112S(n-4)-64S(n-5)   for n>=6
    """
    # Compute small exact values to seed the recurrence and run statement asserts.
    # 2^15 = 32768 terms, fast enough and gives a safety margin for verification.
    S_exact = exact_S_up_to(15)

    # Asserts from the problem statement:
    assert t(1) == 0
    assert t(2) == 0
    assert t(3) == 2
    assert S_exact[10] == 361522

    # Extra sanity: check the recurrence against exact values on a small window.
    for n in range(6, 16):
        lhs = S_exact[n]
        rhs = (
            7 * S_exact[n - 1]
            - 6 * S_exact[n - 2]
            - 48 * S_exact[n - 3]
            + 112 * S_exact[n - 4]
            - 64 * S_exact[n - 5]
        )
        assert lhs == rhs

    if N <= 5:
        return S_exact[N] % mod

    # Seed S(1..5)
    seed = [S_exact[i] % mod for i in range(1, 6)]  # [S1,S2,S3,S4,S5]

    # Maintain a rolling window [S(n-1), S(n-2), S(n-3), S(n-4), S(n-5)]
    prev = [seed[4], seed[3], seed[2], seed[1], seed[0]]  # [S5,S4,S3,S2,S1]

    for n in range(6, N + 1):
        new = (
            7 * prev[0] - 6 * prev[1] - 48 * prev[2] + 112 * prev[3] - 64 * prev[4]
        ) % mod
        prev = [new, prev[0], prev[1], prev[2], prev[3]]

    return prev[0]


def main() -> None:
    ans = S_mod_large(TARGET_N, MOD)
    print(ans)


if __name__ == "__main__":
    main()
