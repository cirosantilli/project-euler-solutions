#!/usr/bin/env python
from __future__ import annotations

from typing import List


def sieve(limit: int) -> bytearray:
    """
    Returns a bytearray is_prime where is_prime[n] is 1 if n is prime else 0,
    for 0 <= n <= limit.
    """
    if limit < 1:
        return bytearray(b"\x00") * (limit + 1)

    is_prime = bytearray(b"\x01") * (limit + 1)
    is_prime[0:2] = b"\x00\x00"
    # Even numbers > 2 are not prime
    for i in range(4, limit + 1, 2):
        is_prime[i] = 0

    p = 3
    while p * p <= limit:
        if is_prime[p]:
            step = 2 * p
            start = p * p
            for x in range(start, limit + 1, step):
                is_prime[x] = 0
        p += 2
    return is_prime


def is_truncatable_prime(n: int, is_prime: bytearray) -> bool:
    # Single-digit primes are explicitly excluded
    if n < 10:
        return False
    s = str(n)
    # Truncate from left and from right; all intermediates must be prime
    for i in range(1, len(s)):
        if not is_prime[int(s[i:])]:
            return False
        if not is_prime[int(s[: len(s) - i])]:
            return False
    return True


def solve() -> int:
    """
    @formalsw: Py from project_euler_statements.p37 import naive_default
    @formalsw: Equiv(naive_default)
    """
    limit = 1000
    while True:
        is_prime = sieve(limit)
        truncatable = []
        for n in range(11, limit + 1, 2):
            if is_prime[n] and is_truncatable_prime(n, is_prime):
                truncatable.append(n)
                if len(truncatable) == 11:
                    return sum(truncatable)
        limit *= 2


if __name__ == "__main__":
    print(solve())
