#!/usr/bin/env python
from __future__ import annotations


def solve(n: int) -> int:
    """
    @formalsw: Py from project_euler_statements.p3 import naive
    @formalsw: Requires(n >= 2)
    @formalsw: Equiv(naive)

    Return the largest prime factor of n.
    """
    largest = 1

    # Factor out 2s
    while n % 2 == 0:
        largest = 2
        n //= 2

    # Factor out odd primes
    f = 3
    while f * f <= n:
        while n % f == 0:
            largest = f
            n //= f
        f += 2

    # If n is now > 1, it's prime and larger than any factor found
    if n > 1:
        largest = n

    return largest


if __name__ == "__main__":
    assert solve(13195) == 29
    print(solve(600851475143))
