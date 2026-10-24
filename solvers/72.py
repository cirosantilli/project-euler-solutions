#!/usr/bin/env python
from __future__ import annotations


def totients_up_to(limit: int) -> list[int]:
    """
    Linear sieve computing phi[0..limit] in O(limit).

    @formalsw: Returns
    """
    phi = [0] * (limit + 1)
    if limit < 1:
        return phi
    is_comp = [False] * (limit + 1)
    primes: list[int] = []

    phi[1] = 1
    for i in range(2, limit + 1):
        if not is_comp[i]:
            primes.append(i)
            phi[i] = i - 1
        for p in primes:
            ip = i * p
            if ip > limit:
                break
            is_comp[ip] = 1
            if i % p == 0:
                phi[ip] = phi[i] * p
                break
            phi[ip] = phi[i] * (p - 1)
    return phi


def solve(limit: int) -> int:
    """
    Count reduced proper fractions with positive denominator at most limit.

    @formalsw: Py from project_euler_statements.p72 import naive
    @formalsw: Returns
    @formalsw: Equiv(naive)
    """
    if limit < 2:
        return 0
    phi = totients_up_to(limit)
    return sum(phi[2:])


if __name__ == "__main__":
    assert solve(8) == 21
    assert solve(1) == 0
    assert solve(2) == 1
    print(solve(1_000_000))
