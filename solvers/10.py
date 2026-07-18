#!/usr/bin/env python

import math


def solve(limit: int) -> int:
    """
    Odd-only sieve of Eratosthenes.
    """
    if limit <= 2:
        return 0
    if limit <= 3:
        # The only prime below 3 is 2
        return 2

    # Store only odd numbers >= 3:
    # index i represents number (2*i + 3)
    size = limit // 2 - 1  # count of odd numbers in [3, limit)
    sieve = bytearray(b"\x01") * size

    r = math.isqrt(limit - 1)
    # largest i such that (2*i+3) <= r
    max_i = (r - 3) // 2

    for i in range(max_i + 1):
        if sieve[i]:
            p = 2 * i + 3
            # index of p*p
            start = (p * p - 3) // 2
            if start < size:
                count = ((size - start - 1) // p) + 1
                sieve[start::p] = b"\x00" * count

    total = 2
    for i, is_prime in enumerate(sieve):
        if is_prime:
            total += 2 * i + 3
    return total


if __name__ == "__main__":
    assert solve(10) == 17
    print(solve(2_000_000))
