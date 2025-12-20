#!/usr/bin/env python3
"""Project Euler 753: Fermat Equation

Compute:
    sum_{prime p < 6_000_000} F(p)

where F(p) counts ordered triples (a,b,c) with 1 <= a,b,c < p satisfying
    a^3 + b^3 ≡ c^3 (mod p).

No external libraries are used.
"""

from __future__ import annotations

import sys
from math import isqrt
from array import array


LIMIT_EXCLUSIVE_DEFAULT = 6_000_000


def sieve_primes_upto(n: int) -> tuple[bytearray, list[int]]:
    """Odd-only sieve up to n (inclusive).

    Returns:
      - sieve_odd: bytearray where sieve_odd[i] is 1 iff (2*i+1) is prime (for i>=1).
                   sieve_odd[0] corresponds to 1 (not prime).
      - primes: list of primes <= n
    """
    if n < 2:
        return bytearray(b"\x00"), []

    size = (n // 2) + 1  # odds up to n
    sieve_odd = bytearray(b"\x01") * size
    sieve_odd[0] = 0  # 1 is not prime

    r = isqrt(n)
    for i in range(1, (r // 2) + 1):
        if sieve_odd[i]:
            p = 2 * i + 1
            start = (p * p) // 2
            sieve_odd[start::p] = b"\x00" * (((size - 1 - start) // p) + 1)

    primes = [2]
    primes.extend(2 * i + 1 for i in range(1, size) if sieve_odd[i])
    if primes and primes[-1] > n:
        primes.pop()
    return sieve_odd, primes


def build_u_map(limit_inclusive: int, sieve_odd: bytearray) -> array:
    """Build u_by_p for primes p ≡ 1 (mod 3) with 4p = u^2 + 27 v^2.

    For the given bound (~6e6), v < 1000, so enumerating (u,v) pairs is fast and
    avoids per-prime modular square roots.
    """
    n = limit_inclusive
    n4 = 4 * n
    vmax = isqrt(n4 // 27)
    u_by_p = array("H", [0]) * (n + 1)

    # v=0 would imply 4p is a square, impossible for prime p.
    for v in range(1, vmax + 1):
        base = 27 * v * v
        umax = isqrt(n4 - base)

        # Ensure (u^2 + base) divisible by 4:
        # base % 4 is 0 if v even else 3; u^2 % 4 is 0 for even u else 1.
        # Thus u parity must match v parity.
        u = v & 1
        while u <= umax:
            p = (u * u + base) >> 2
            if p <= n and (p % 3 == 1):
                if p != 2 and (p & 1) and sieve_odd[p >> 1]:
                    if u_by_p[p] == 0:
                        u_by_p[p] = u
            u += 2
    return u_by_p


def trace_ap_for_curve(p: int, u_by_p: array) -> int:
    """Trace a_p for the CM elliptic curve isomorphic to X^3 + Y^3 = Z^3 (p != 3).

    - If p ≡ 2 (mod 3): a_p = 0.
    - If p ≡ 1 (mod 3): find u with 4p = u^2 + 27 v^2, then |a_p| = u.
      The curve has a rational 3-torsion point, so #E(F_p) is divisible by 3,
      forcing a_p ≡ p+1 (mod 3). For p ≡ 1 (mod 3), that means a_p ≡ 2 (mod 3),
      which fixes the sign.
    """
    if p % 3 == 2:
        return 0
    if p % 3 != 1:
        raise ValueError("p must be 1 or 2 mod 3 (and not 3)")
    u = u_by_p[p]
    if u == 0:
        raise RuntimeError(f"Missing u for prime p={p}")
    return u if (u % 3) == 2 else -u


def F_of_prime(p: int, u_by_p: array) -> int:
    """Compute F(p) for a prime p."""
    if p == 3:
        # Small special case: direct count for p=3.
        cubes = [0, 1, 2]  # x^3 mod 3 for x=0,1,2
        cnt = 0
        for a in (1, 2):
            a3 = cubes[a]
            for b in (1, 2):
                s = (a3 + cubes[b]) % 3
                for c in (1, 2):
                    if cubes[c] == s:
                        cnt += 1
        return cnt

    if p % 3 == 2:
        # Cube map is a bijection on F_p^*.
        return (p - 1) * (p - 2)

    # p ≡ 1 (mod 3)
    ap = trace_ap_for_curve(p, u_by_p)
    # For p ≡ 1 (mod 3):
    #   F(p) = (p-1) * (#E(F_p) - 9)
    # and #E(F_p) = p + 1 - a_p.
    return (p - 1) * (p - ap - 8)


def solve(limit_exclusive: int = LIMIT_EXCLUSIVE_DEFAULT) -> int:
    """Return sum_{prime p < limit_exclusive} F(p)."""
    if limit_exclusive <= 2:
        return 0

    max_p = limit_exclusive - 1
    sieve_odd, primes = sieve_primes_upto(max_p)
    u_by_p = build_u_map(max_p, sieve_odd)

    # Problem statement checks:
    assert F_of_prime(5, u_by_p) == 12
    assert F_of_prime(7, u_by_p) == 0

    total = 0
    for p in primes:
        if p >= limit_exclusive:
            break
        total += F_of_prime(p, u_by_p)
    return total


def main(argv: list[str]) -> None:
    if len(argv) >= 2:
        limit_exclusive = int(argv[1].replace("_", ""))
    else:
        limit_exclusive = LIMIT_EXCLUSIVE_DEFAULT
    print(solve(limit_exclusive))


if __name__ == "__main__":
    main(sys.argv)
