#!/usr/bin/env python
from __future__ import annotations

import math


def initial_limit(n: int) -> int:
    """
    @formalsw: Returns
    """
    if n < 6:
        return 15
    log2_n = n.bit_length() - 1
    return n * (log2_n + (log2_n + 1).bit_length()) + 10


def solve(n: int) -> int:
    """
    @formalsw: Py from project_euler_statements.p7 import naive
    @formalsw: Requires(n >= 1)
    @formalsw: Equiv(naive)
    """
    assert n >= 1
    if n == 1:
        return 2

    # An integer starting estimate avoids float overflow for large n.
    # The retry loop grows it if the sieve contains too few primes.
    limit = initial_limit(n)

    while True:
        # Sieve odds only: index i represents number (2*i + 1)
        size = (limit // 2) + 1
        is_prime = bytearray(b"\x01") * size
        is_prime[0] = 0  # 1 is not prime

        root = int(math.isqrt(limit))
        # i corresponds to p = 2*i + 1
        max_i = root // 2
        for i in range(1, max_i + 1):
            if is_prime[i]:
                p = 2 * i + 1
                start = (p * p) // 2
                step = p
                count = ((size - 1 - start) // step) + 1
                is_prime[start::step] = b"\x00" * count

        # Count primes and stop at the nth
        count_primes = 1  # prime = 2
        for i in range(1, size):
            if is_prime[i]:
                count_primes += 1
                if count_primes == n:
                    return 2 * i + 1

        # Not enough primes; increase limit and retry
        limit *= 2


if __name__ == "__main__":
    assert solve(1) == 2
    assert solve(6) == 13
    print(solve(10001))
