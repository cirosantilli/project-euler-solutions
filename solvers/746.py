#!/usr/bin/env python3
"""
Project Euler 746 - A Messy Dinner
Compute S(2021) = sum_{k=2..2021} M(k) (mod 1_000_000_007).

No external libraries are used.
"""

import sys

MOD = 1_000_000_007


def build_factorials(n_max: int):
    """Precompute factorials and inverse factorials up to n_max (inclusive) modulo MOD."""
    fact = [1] * (n_max + 1)
    for i in range(1, n_max + 1):
        fact[i] = (fact[i - 1] * i) % MOD
    invfact = [1] * (n_max + 1)
    invfact[n_max] = pow(fact[n_max], MOD - 2, MOD)
    for i in range(n_max, 0, -1):
        invfact[i - 1] = (invfact[i] * i) % MOD
    return fact, invfact


def nCk(n: int, k: int, fact, invfact) -> int:
    if k < 0 or k > n:
        return 0
    return fact[n] * invfact[k] % MOD * invfact[n - k] % MOD


def precompute_inverses(n_max: int):
    """Compute modular inverses of 1..n_max modulo MOD in O(n)."""
    inv = [0] * (n_max + 1)
    inv[1] = 1
    for i in range(2, n_max + 1):
        inv[i] = MOD - (MOD // i) * inv[MOD % i] % MOD
    return inv


def M(n: int, fact, invfact, inv, pow4) -> int:
    """
    Count M(n) modulo MOD.

    We count seatings on labelled circular seats (rotation is NOT factored out),
    under two possible alternating gender patterns. We compute for the pattern
    where even-indexed seats are male and odd-indexed seats are female, then
    multiply by 2 for the two patterns.

    Special case n=1: every seating has the single family together.
    """
    if n <= 0:
        return 0
    if n == 1:
        return 0

    total = 0

    # falling factorial P(n,k) = n!/(n-k)! computed incrementally
    nPk = 1

    for k in range(0, n + 1):
        if k > 0:
            nPk = (nPk * (n - (k - 1))) % MOD

        if k == 0:
            D = 1
        else:
            # Number of ways to choose k disjoint length-4 intervals on a cycle of length 4n
            # (unlabelled intervals):
            #   D = (4n / k) * C(4n - 3k - 1, k - 1)
            # (valid for n>1; n==1 is handled above).
            D = (4 * n) % MOD
            D = (D * inv[k]) % MOD
            D = (D * nCk(4 * n - 3 * k - 1, k - 1, fact, invfact)) % MOD

        rem = 2 * (n - k)  # remaining men and women each
        ways_rest = fact[rem] * fact[rem] % MOD
        term = nPk
        term = (term * D) % MOD
        term = (term * pow4[k]) % MOD
        term = (term * ways_rest) % MOD

        if k & 1:
            total = (total - term) % MOD
        else:
            total = (total + term) % MOD

    # Two possible alternating gender patterns around the circle
    return (2 * total) % MOD


def S(n: int, fact, invfact, inv, pow4) -> int:
    """S(n) = sum_{k=2..n} M(k) modulo MOD."""
    acc = 0
    for k in range(2, n + 1):
        acc += M(k, fact, invfact, inv, pow4)
        acc %= MOD
    return acc


def main() -> None:
    target = 2021
    if len(sys.argv) >= 2:
        try:
            target = int(sys.argv[1])
        except ValueError:
            target = 2021

    # We must support asserts for M(10), S(10) from the statement.
    max_n = max(target, 10)

    # Need factorials up to 4*max_n and inverses up to max_n.
    fact, invfact = build_factorials(4 * max_n)
    inv = precompute_inverses(max_n)
    pow4 = [1] * (max_n + 1)
    for i in range(1, max_n + 1):
        pow4[i] = (pow4[i - 1] * 4) % MOD

    # Test values from the problem statement
    assert M(1, fact, invfact, inv, pow4) == 0
    assert M(2, fact, invfact, inv, pow4) == 896
    assert M(3, fact, invfact, inv, pow4) == 890880
    assert M(10, fact, invfact, inv, pow4) % MOD == 170717180
    assert S(10, fact, invfact, inv, pow4) % MOD == 399291975

    print(S(target, fact, invfact, inv, pow4) % MOD)


if __name__ == "__main__":
    main()
