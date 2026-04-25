#!/usr/bin/env python
"""
Project Euler 501: Eight Divisors

Count integers n <= 10^12 with exactly 8 divisors.

The only possible prime-exponent patterns are p^7, p^3*q with p != q,
and p*q*r with distinct primes.  The implementation builds a prime-count
table for the fixed limit, so every needed pi(x) query is an array lookup.
"""

from bisect import bisect_right
from math import isqrt


N = 10**12


def iroot(n: int, k: int) -> int:
    """Return floor(n ** (1/k)), corrected around floating-point boundaries."""
    if n < 2:
        return n
    r = int(round(n ** (1.0 / k)))
    while (r + 1) ** k <= n:
        r += 1
    while r**k > n:
        r -= 1
    return r


def sieve_primes(limit: int) -> list[int]:
    if limit < 2:
        return []

    flags = bytearray(b"\x01") * (limit + 1)
    flags[0:2] = b"\x00\x00"
    for p in range(2, isqrt(limit) + 1):
        if flags[p]:
            start = p * p
            flags[start : limit + 1 : p] = b"\x00" * (((limit - start) // p) + 1)
    return [i for i in range(2, limit + 1) if flags[i]]


class PrimeCounter:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.root = isqrt(limit)
        self.primes = sieve_primes(self.root)

        # small[x] tracks pi(x) for x <= sqrt(limit).  large[d] tracks
        # pi(limit // d), which covers every larger query used below.
        root = self.root
        small = [0] * (root + 1)
        for x in range(2, root + 1):
            small[x] = x - 1
        large = [0] * (root + 1)
        for d in range(1, root + 1):
            large[d] = limit // d - 1

        for p in self.primes:
            p2 = p * p
            if p2 > limit:
                break
            before_p = small[p - 1]

            last_d = min(root, limit // p2)
            for d in range(1, last_d + 1):
                pd = p * d
                if pd <= root:
                    large[d] -= large[pd] - before_p
                else:
                    large[d] -= small[limit // pd] - before_p

            for x in range(root, p2 - 1, -1):
                small[x] -= small[x // p] - before_p

        self.small = small
        self.large = large

    def pi(self, x: int) -> int:
        if x < 2:
            return 0
        if x <= self.root:
            return self.small[x]
        return self.large[self.limit // x]


def count_eight_divisors(limit: int) -> int:
    if limit < 24:
        return 0

    counter = PrimeCounter(limit)
    primes = counter.primes

    count = 0

    # p*q*r, with p < q < r.
    p_stop = bisect_right(primes, iroot(limit, 3))
    for i in range(p_stop):
        p = primes[i]
        if p * p * p >= limit:
            break
        q_stop = bisect_right(primes, isqrt(limit // p))
        for j in range(i + 1, q_stop):
            q = primes[j]
            max_r = limit // (p * q)
            if max_r <= q:
                break
            count += counter.pi(max_r) - (j + 1)

    # Raw p^3*q count, then remove the forbidden q = p cases.
    for p in primes[:p_stop]:
        p3 = p * p * p
        if p3 > limit:
            break
        count += counter.pi(limit // p3)
    count -= counter.pi(iroot(limit, 4))

    # p^7.
    count += counter.pi(iroot(limit, 7))
    return count


def main() -> None:
    assert count_eight_divisors(100) == 10
    assert count_eight_divisors(1000) == 180
    assert count_eight_divisors(10**6) == 224427

    print(count_eight_divisors(N))


if __name__ == "__main__":
    main()
