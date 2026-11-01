#!/usr/bin/env python
from __future__ import annotations

from math import isqrt


def sieve_primes(n: int) -> list[int]:
    """
    Return the primes at most n.

    @formalsw: Returns
    """
    if n < 2:
        return []
    is_prime = bytearray(b"\x01") * (n + 1)
    is_prime[0:2] = b"\x00\x00"
    limit = isqrt(n)
    for p in range(2, limit + 1):
        if is_prime[p]:
            step = p
            start = p * p
            is_prime[start : n + 1 : step] = b"\x00" * (((n - start) // step) + 1)
    return [i for i in range(2, n + 1) if is_prime[i]]


def solve(limit: int) -> int:
    """
    Count numbers < limit expressible as p^2 + q^3 + r^4 with p,q,r prime.

    @formalsw: Py from project_euler_statements.p87 import naive
    @formalsw: Returns
    @formalsw: Equiv(naive)
    """
    if limit <= 0:
        return 0

    # Need primes up to sqrt(limit) to cover all squares.
    primes = sieve_primes(isqrt(limit) + 1)

    squares: list[int] = []
    cubes: list[int] = []
    fourths: list[int] = []

    for p in primes:
        p2 = p * p
        if p2 < limit:
            squares.append(p2)
        p3 = p2 * p
        if p3 < limit:
            cubes.append(p3)
        p4 = p2 * p2
        if p4 < limit:
            fourths.append(p4)

    squares.sort()
    cubes.sort()
    fourths.sort()

    representable = set()

    for f in fourths:
        for c in cubes:
            fc = f + c
            if fc >= limit:
                break
            for s in squares:
                total = fc + s
                if total >= limit:
                    break
                representable.add(total)

    return len(representable)


if __name__ == "__main__":
    # From the problem statement: exactly four numbers below 50 are representable.
    assert solve(50) == 4

    print(solve(50_000_000))
