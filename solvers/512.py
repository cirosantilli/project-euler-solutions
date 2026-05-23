#!/usr/bin/env python
"""
Project Euler 512: Sums of Totients of Powers

The required value is the prefix sum of phi(m) over odd m.  A divisor identity
relates it to the summatory odd part of integers, giving a fast memoized
quotient-block recurrence.
"""

from functools import cache


TARGET = 500_000_000


def odd_part_sum(n: int) -> int:
    """Return sum_{1<=x<=n} odd_part(x)."""
    total = 0
    while n:
        odd_count = (n + 1) // 2
        total += odd_count * odd_count
        n //= 2
    return total


@cache
def odd_totient_prefix(n: int) -> int:
    """Return sum(phi(m) for odd m <= n)."""
    total = odd_part_sum(n)

    lo = 2
    while lo <= n:
        q = n // lo
        hi = n // q
        total -= (hi - lo + 1) * odd_totient_prefix(q)
        lo = hi + 1

    return total


def totient_sieve(n: int) -> int:
    """Small direct checker for odd totient prefixes."""
    phi = list(range(n + 1))
    for p in range(2, n + 1):
        if phi[p] == p:
            for multiple in range(p, n + 1, p):
                phi[multiple] -= phi[multiple] // p
    return sum(phi[x] for x in range(1, n + 1, 2))


def main() -> None:
    assert odd_part_sum(10) == 36
    assert odd_totient_prefix(10) == 19
    assert odd_totient_prefix(1000) == totient_sieve(1000)
    print(odd_totient_prefix(TARGET))


if __name__ == "__main__":
    main()
