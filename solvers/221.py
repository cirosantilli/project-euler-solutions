#!/usr/bin/env python
"""Project Euler 221: Alexandrian Integers."""

from __future__ import annotations

import heapq
import math


def sieve_primes(limit: int) -> list[int]:
    if limit < 2:
        return []
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[0:2] = b"\x00\x00"
    for p in range(2, math.isqrt(limit) + 1):
        if sieve[p]:
            sieve[p * p : limit + 1 : p] = b"\x00" * (((limit - p * p) // p) + 1)
    return [i for i in range(2, limit + 1) if sieve[i]]


def factor_with_primes(n: int, primes: list[int]) -> list[tuple[int, int]]:
    factors: list[tuple[int, int]] = []
    rem = n
    for p in primes:
        if p * p > rem:
            break
        if rem % p:
            continue
        exp = 0
        while rem % p == 0:
            rem //= p
            exp += 1
        factors.append((p, exp))
    if rem > 1:
        factors.append((rem, 1))
    return factors


def divisors_up_to_root(factors: list[tuple[int, int]], root: int) -> list[int]:
    divisors = [1]
    for p, exp in factors:
        base = list(divisors)
        power = 1
        for _ in range(exp):
            power *= p
            for d in base:
                value = d * power
                if value <= root:
                    divisors.append(value)
    return divisors


def compute(k: int = 150000) -> int:
    prime_limit = 200000
    primes = sieve_primes(prime_limit)

    # Python has a min-heap, so store negative values to keep the current k
    # smallest Alexandrian integers as a bounded max-heap.
    heap: list[int] = []
    active: set[int] = set()

    p = 1
    while True:
        if p >= prime_limit:
            prime_limit *= 2
            primes = sieve_primes(prime_limit)

        n = p * p + 1
        factors = factor_with_primes(n, primes)
        root = math.isqrt(n)

        for d in divisors_up_to_root(factors, root):
            e = n // d
            value = p * (p + d) * (p + e)
            if value in active:
                continue
            if len(heap) < k:
                heapq.heappush(heap, -value)
                active.add(value)
            elif value < -heap[0]:
                removed = -heapq.heapreplace(heap, -value)
                active.remove(removed)
                active.add(value)

        p += 1
        if len(heap) == k and p * (p + 1) * (p + 1) > -heap[0]:
            return -heap[0]


def main() -> None:
    assert compute(6) == 630
    print(compute())


if __name__ == "__main__":
    main()
