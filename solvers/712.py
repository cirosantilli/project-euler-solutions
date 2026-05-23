#!/usr/bin/env python
"""
Project Euler 712 - Exponent Difference

For each prime power p^k, the ordered pairs with different divisibility by p^k
contribute 2*q*(N-q), where q = floor(N/p^k).  Small prime powers are generated
directly; the remaining large primes are grouped by the common quotient
floor(N/p).
"""

from math import isqrt


MOD = 10**9 + 7
TARGET = 10**12
SMALL_PRIME_LIMIT = 10**8
SEGMENT_ODD_COUNT = 1_000_000


def simple_primes(limit: int) -> list[int]:
    if limit < 2:
        return []
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[0:2] = b"\x00\x00"
    for p in range(2, isqrt(limit) + 1):
        if sieve[p]:
            sieve[p * p : limit + 1 : p] = b"\x00" * (((limit - p * p) // p) + 1)
    return [n for n in range(limit + 1) if sieve[n]]


def segmented_primes(limit: int):
    if limit >= 2:
        yield 2
    if limit < 3:
        return

    base_primes = simple_primes(isqrt(limit))
    span = 2 * SEGMENT_ODD_COUNT
    low = 3
    while low <= limit:
        high = min(limit, low + span - 2)
        size = (high - low) // 2 + 1
        block = bytearray(b"\x01") * size

        for p in base_primes[1:]:
            if p * p > high:
                break
            start = max(p * p, ((low + p - 1) // p) * p)
            if start % 2 == 0:
                start += p
            index = (start - low) // 2
            block[index::p] = b"\x00" * (((size - 1 - index) // p) + 1)

        for i, is_prime in enumerate(block):
            if is_prime:
                yield low + 2 * i

        low += span


def prime_count_tables(n: int) -> tuple[int, list[int], list[int]]:
    """
    Return pi(x) tables for x <= sqrt(n) and x = floor(n/i).

    This is the same compressed-value sieve used by many prime-counting
    routines: initialize counts of integers >= 2, then remove composites by
    smallest prime factor.
    """
    root = isqrt(n)
    primes = simple_primes(root)

    values = [0] * (root + 1)
    small = [0] * (root + 1)
    large = [0] * (root + 1)

    for x in range(2, root + 1):
        small[x] = x - 1
    for i in range(1, root + 1):
        value = n // i
        values[i] = value
        large[i] = value - 1

    for p in primes:
        p2 = p * p
        if p2 > n:
            break
        before = small[p - 1]

        last = min(root, n // p2)
        for i in range(1, last + 1):
            y = values[i] // p
            large[i] -= (small[y] if y <= root else large[n // y]) - before

        for x in range(root, p2 - 1, -1):
            small[x] -= small[x // p] - before

    return root, small, large


def D(n: int, m: int) -> int:
    total = 0
    for p in simple_primes(max(n, m)):
        vn = 0
        vm = 0
        nn = n
        mm = m
        while nn % p == 0:
            nn //= p
            vn += 1
        while mm % p == 0:
            mm //= p
            vm += 1
        total += abs(vn - vm)
    return total


def S_small(n: int) -> int:
    return sum(D(a, b) for a in range(1, n + 1) for b in range(1, n + 1))


def contribution(quotient: int, n: int, mod: int) -> int:
    return (quotient % mod) * ((n - quotient) % mod) % mod


def S(n: int = TARGET, mod: int = MOD) -> int:
    if n < 2:
        return 0

    cutoff = min(n, SMALL_PRIME_LIMIT)
    total = 0

    for p in segmented_primes(cutoff):
        power = p
        while power <= n:
            q = n // power
            total = (total + contribution(q, n, mod)) % mod
            if power > n // p:
                break
            power *= p

    if cutoff < n:
        root, pi_small, pi_large = prime_count_tables(n)

        def pi(x: int) -> int:
            if x <= root:
                return pi_small[x]
            return pi_large[n // x]

        max_q = n // (cutoff + 1)
        for q in range(1, max_q + 1):
            hi = n // q
            lo_exclusive = n // (q + 1)
            if lo_exclusive < cutoff:
                lo_exclusive = cutoff
            if hi <= lo_exclusive:
                continue
            count = pi(hi) - pi(lo_exclusive)
            total = (total + count * contribution(q, n, mod)) % mod

    return 2 * total % mod


if __name__ == "__main__":
    assert D(14, 24) == 4
    assert S_small(10) == 210
    assert S_small(100) == 37018
    print(S())
