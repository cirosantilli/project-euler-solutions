#!/usr/bin/env python
from __future__ import annotations

from math import isqrt


def qualifying_numbers(limit: int) -> list[int]:
    if limit <= 1:
        return []
    bound = isqrt(isqrt(limit - 1))
    sieve = bytearray(b"\x01") * (bound + 1)
    sieve[:2] = b"\x00\x00"
    for p in range(2, isqrt(bound) + 1):
        if sieve[p]:
            sieve[p * p : bound + 1 : p] = b"\x00" * (
                (bound - p * p) // p + 1
            )
    primes = [p for p in range(2, bound + 1) if sieve[p]]
    result = []
    # tau(n) = 25 permits exactly p^24 or p^4 q^4, with distinct primes.
    for p in primes:
        value = p**24
        if value >= limit:
            break
        if sum(p**k for k in range(25)) % 25 == 0:
            result.append(value)

    # sigma(p^4) is divisible by 5 iff p = 1 mod 5, and is then 5 mod 25.
    candidates = [p for p in primes if p % 5 == 1]
    for i, p in enumerate(candidates):
        for q in candidates[i + 1 :]:
            if p * q > bound:
                break
            result.append((p * q) ** 4)
    return result


def main() -> None:
    first = 13521270961
    assert qualifying_numbers(first) == []
    assert qualifying_numbers(first + 1) == [first]
    print(sum(qualifying_numbers(10**15)))


if __name__ == "__main__":
    main()
