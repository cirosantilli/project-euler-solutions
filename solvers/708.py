#!/usr/bin/env python3
"""
Project Euler 708: Twos Are All You Need

We need S(N) = sum_{n<=N} 2^{Omega(n)}.
"""

from __future__ import annotations

from array import array
from math import isqrt
from typing import Dict, List


def primes_upto(limit: int) -> List[int]:
    if limit < 2:
        return []
    size = limit // 2 + 1  # odd numbers only
    sieve = bytearray(b"\x01") * size
    sieve[0] = 0
    r = isqrt(limit)
    for p in range(3, r + 1, 2):
        if sieve[p // 2]:
            start = p * p
            step = p
            sieve[start // 2 :: step] = b"\x00" * ((size - start // 2 - 1) // step + 1)
    primes = [2]
    primes.extend(2 * i + 1 for i in range(1, size) if sieve[i])
    return primes


def build_divisor_prefix(limit: int) -> array:
    spf = array("I", [0]) * (limit + 1)
    for i in range(2, limit + 1):
        if spf[i] == 0:
            spf[i] = i
            if i * i <= limit:
                for j in range(i * i, limit + 1, i):
                    if spf[j] == 0:
                        spf[j] = i

    d = array("I", [0]) * (limit + 1)
    exp = array("I", [0]) * (limit + 1)
    d[1] = 1
    for i in range(2, limit + 1):
        p = spf[i]
        m = i // p
        if m % p == 0:
            exp[i] = exp[m] + 1
            d[i] = d[m] // (exp[m] + 1) * (exp[i] + 1)
        else:
            exp[i] = 1
            d[i] = d[m] * 2

    prefix = array("Q", [0]) * (limit + 1)
    total = 0
    for i in range(1, limit + 1):
        total += d[i]
        prefix[i] = total
    return prefix


def sum_twos(
    n: int, primes: List[int], prime_sq: List[int], prefix: array, small_limit: int
) -> int:
    cache: Dict[int, int] = {}

    def D(x: int) -> int:
        if x <= small_limit:
            return prefix[x]
        cached = cache.get(x)
        if cached is not None:
            return cached
        s = isqrt(x)
        total = 0
        i = 1
        while i <= s:
            q = x // i
            j = x // q
            if j > s:
                j = s
            total += q * (j - i + 1)
            i = j + 1
        res = 2 * total - s * s
        cache[x] = res
        return res

    primes_local = primes
    prime_sq_local = prime_sq
    D_local = D
    total = 0

    def dfs(start_idx: int, current_n: int, current_g: int) -> None:
        nonlocal total
        total += current_g * D_local(n // current_n)
        limit = n // current_n
        for i in range(start_idx, len(primes_local)):
            p2 = prime_sq_local[i]
            if p2 > limit:
                break
            p = primes_local[i]
            pow_p = p2
            g = current_g
            while pow_p <= limit:
                dfs(i + 1, current_n * pow_p, g)
                pow_p *= p
                g <<= 1

    dfs(0, 1, 1)
    return total


def main() -> None:
    max_n = 10**14
    prime_limit = isqrt(max_n)
    primes = primes_upto(prime_limit)
    prime_sq = [p * p for p in primes]

    small_limit = 1_000_000
    prefix = build_divisor_prefix(small_limit)

    assert sum_twos(10**8, primes, prime_sq, prefix, small_limit) == 9613563919
    print(sum_twos(10**14, primes, prime_sq, prefix, small_limit))


if __name__ == "__main__":
    main()
