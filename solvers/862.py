#!/usr/bin/env python3
"""
Project Euler 862: Larger Digit Permutation

For a positive integer n, define T(n) as the number of strictly larger integers
that can be formed by permuting the digits of n (leading zeros not allowed).
Define S(k) = sum of T(n) over all k-digit numbers n.

This program computes S(12) combinatorially (no enumeration of 12-digit numbers).

No third-party libraries are used.
"""

from __future__ import annotations

from itertools import permutations
from typing import List


def factorials(n: int) -> List[int]:
    """Return [0!, 1!, ..., n!]."""
    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = fact[i - 1] * i
    return fact


def T_bruteforce(n: int) -> int:
    """Brute force T(n) by enumerating all digit permutations (small tests only)."""
    ds = list(str(n))
    vals = set()
    for p in permutations(ds):
        if p[0] == "0":
            continue
        vals.add(int("".join(p)))
    return sum(1 for v in vals if v > n)


def S(k: int) -> int:
    """Compute S(k) using digit-multiset enumeration.

    Group k-digit numbers by digit multiplicities d0..d9 (sum di = k).

    For fixed multiplicities, let M be the number of *distinct* valid k-digit
    permutations (no leading zero). For any such group, summing T(n) over all
    group elements equals C(M,2), because each unordered pair contributes exactly
    one larger/smaller relation.

    We enumerate all multiplicity vectors and add M*(M-1)/2.
    """
    if k <= 1:
        return 0

    fact = factorials(k)
    fk1 = fact[k - 1]  # (k-1)!
    total = 0

    # Choose d0 (count of zeros). Remaining r = k-d0 digits belong to {1..9}.
    for d0 in range(0, k + 1):
        r = k - d0
        if r == 0:
            continue  # all zeros -> cannot make a k-digit number

        denom0 = fact[d0]  # contributes d0! to denominator

        # Recursively assign counts for digits 1..9 summing to r.
        # denom_prod accumulates Π_{i=1..9} di!
        def rec(digit: int, rem: int, denom_prod: int) -> None:
            nonlocal total
            if digit == 10:
                if rem != 0:
                    return
                denom = denom0 * denom_prod

                # Number of distinct valid permutations (leading non-zero):
                # M = (k! / Π di!) - ((k-1)! / ((d0-1)! Π_{i>0} di!))
                #   = r * (k-1)! / Π di!
                M = (r * fk1) // denom
                total += M * (M - 1) // 2
                return

            if digit == 9:
                # Last digit forced to take whatever remains.
                c = rem
                rec(10, 0, denom_prod * fact[c])
                return

            for c in range(rem + 1):
                rec(digit + 1, rem - c, denom_prod * fact[c])

        rec(1, r, 1)

    return total


def main() -> None:
    # Test values from the problem statement
    assert T_bruteforce(2302) == 4
    assert S(3) == 1701

    print(S(12))


if __name__ == "__main__":
    main()
