#!/usr/bin/env python3
"""Project Euler 817

Define m = M(n, d) to be the smallest positive integer such that when m^2 is
written in base n it contains the digit d.

We must compute:

    S = sum_{d=1}^{100000} M(p, p - d)

where p = 1_000_000_007.

Key ideas

For each d we want the digit a = p-d (i.e. a ≡ -d mod p) to appear somewhere in
the base-p expansion of m^2.

1) Least-significant digit (units in base p)
   If the digit a appears as the least-significant base-p digit of m^2 then
   we need m^2 ≡ a ≡ -d (mod p). Because p is prime and p % 4 == 3, we can:
     - test if -d is a quadratic residue with Euler's criterion,
     - if it is, compute the modular square root using a^{(p+1)/4} (mod p).
   The smaller of the two roots gives M(p, p-d) in this case.

2) Next digit (the p's place)
   If -d is a non-residue, the least-significant digit can never be a for any
   integer m, so the first occurrence must be in a higher digit. For the digits
   we ask for (a = p-d with d <= 1e5 and huge p), the first occurrence happens
   in the next digit. This reduces to finding the smallest m such that m^2 falls
   into an interval of width p near a multiple of p^2:

       [B*p^2 - d*p,  B*p^2 - (d-1)*p - 1]    for some integer B >= 1.

   For increasing B, we take m = ceil(sqrt(L)) for the lower bound L, and check
   whether m^2 stays inside the width-p window.

The example values given in the statement are asserted using a tiny brute-force
routine (fast enough for base 10/11).

No external libraries are used.
"""

from __future__ import annotations

import math
import sys


def _contains_digit_in_base(x: int, base: int, digit: int) -> bool:
    """Return True iff x written in base `base` contains the digit `digit`."""
    if digit < 0 or digit >= base:
        return False
    while x:
        if x % base == digit:
            return True
        x //= base
    return digit == 0  # x == 0


def _M_bruteforce(n: int, d: int) -> int:
    """Brute-force M(n,d) (only used for statement examples)."""
    m = 1
    while True:
        if _contains_digit_in_base(m * m, n, d):
            return m
        m += 1


def _ceil_isqrt(n: int) -> int:
    """Return ceil(sqrt(n)) for n >= 0."""
    if n <= 0:
        return 0
    # ceil(sqrt(n)) == isqrt(n-1) + 1 for n > 0
    return math.isqrt(n - 1) + 1


def _M_for_prime_base_p(p: int, d: int, exp_leg: int, exp_sqrt: int, p2: int) -> int:
    """Compute M(p, p-d) for prime p with p % 4 == 3, and 1 <= d < p."""
    a = p - d  # a ≡ -d (mod p)

    # Euler's criterion for Legendre symbol:
    #   a^((p-1)/2) ≡  1 (mod p) if a is a residue,
    #             ≡ -1 (mod p) if a is a non-residue.
    ls = pow(a, exp_leg, p)

    if ls == 1:
        # Modular square root for primes p ≡ 3 (mod 4): sqrt(a) = a^((p+1)/4) (mod p)
        r = pow(a, exp_sqrt, p)
        r = r if r <= p - r else p - r
        return r

    # Non-residue: search for first occurrence in the p's place.
    dp = d * p
    L = p2 - dp  # B = 1

    while True:
        m = _ceil_isqrt(L)
        if m * m <= L + (p - 1):
            return m
        L += p2  # increment B


def solve(D: int = 100_000) -> int:
    """Compute sum_{d=1..D} M(p, p-d) for p = 1_000_000_007."""
    p = 1_000_000_007
    if p % 4 != 3:
        raise ValueError("This implementation assumes p % 4 == 3")

    exp_leg = (p - 1) // 2
    exp_sqrt = (p + 1) // 4
    p2 = p * p

    total = 0
    for d in range(1, D + 1):
        total += _M_for_prime_base_p(p, d, exp_leg, exp_sqrt, p2)
    return total


def main() -> None:
    # Statement examples
    assert _M_bruteforce(10, 7) == 24
    assert _M_bruteforce(11, 10) == 19

    D = 100_000
    if len(sys.argv) >= 2:
        D = int(sys.argv[1])
        if D < 1:
            raise SystemExit("D must be >= 1")

    print(solve(D))


if __name__ == "__main__":
    main()
