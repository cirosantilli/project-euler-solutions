#!/usr/bin/env python
"""
Project Euler 642: Sum of largest prime factors.

The solver follows the Min_25-style prime-sum table plus sparse prime-power
recursion described in the accompanying explanation.
"""

from math import isqrt

MOD = 10**9
FLOAT_SQRT_SAFE_LIMIT = 1 << 53


def fast_isqrt(n: int) -> int:
    """Fast floor sqrt for this problem's n < 2^53 recursion bounds."""
    return int(n**0.5)


def sieve_primes(limit: int) -> list[int]:
    if limit < 2:
        return []

    size = limit // 2 + 1
    sieve = bytearray(b"\x01") * size
    sieve[0] = 0
    root = isqrt(limit)
    for i in range(1, root // 2 + 1):
        if sieve[i]:
            p = 2 * i + 1
            start = p * p // 2
            sieve[start::p] = b"\x00" * (((size - 1 - start) // p) + 1)

    primes = [2]
    primes.extend(2 * i + 1 for i in range(1, size) if sieve[i] and 2 * i + 1 <= limit)
    return primes


def build_prime_sum_table(N: int):
    """
    Build prime sums modulo MOD on the usual distinct floor-division domain.
    """
    root = isqrt(N)
    large = [N // i for i in range(1, root + 1)]
    values = large + list(range(large[-1] - 1, 0, -1))

    idx_small = [0] * (root + 1)
    idx_large = [0] * (root + 1)
    for idx, value in enumerate(values):
        if value <= root:
            idx_small[value] = idx
        else:
            idx_large[N // value] = idx

    prime_sums = [((value * (value + 1) // 2) - 1) % MOD for value in values]
    primes = sieve_primes(root)

    limit = len(values)
    for p in primes:
        p2 = p * p
        while limit > 0 and values[limit - 1] < p2:
            limit -= 1

        sum_before_p = prime_sums[idx_small[p - 1]]
        for idx in range(limit):
            value = values[idx]
            quotient = value // p
            qidx = idx_small[quotient] if quotient <= root else idx_large[N // quotient]
            prime_sums[idx] = (
                prime_sums[idx] - p * (prime_sums[qidx] - sum_before_p)
            ) % MOD

    return root, idx_small, idx_large, prime_sums, primes


def compute(N: int) -> int:
    assert N < FLOAT_SQRT_SAFE_LIMIT
    root, idx_small, idx_large, prime_sums, primes = build_prime_sum_table(N)
    prime_count = len(primes)
    key_base = prime_count + 1

    pi = [0] * (root + 1)
    count = 0
    prime_idx = 0
    for value in range(root + 1):
        if prime_idx < prime_count and primes[prime_idx] == value:
            count += 1
            prime_idx += 1
        pi[value] = count

    lower_prime_sum = [0] * (prime_count + 1)
    for idx, p in enumerate(primes):
        lower_prime_sum[idx] = prime_sums[idx_small[p - 1]]
    lower_prime_sum[prime_count] = prime_sums[idx_small[primes[-1]]]

    def prime_sum_upto(x: int) -> int:
        if x <= root:
            return prime_sums[idx_small[x]]
        return prime_sums[idx_large[N // x]]

    def terminal_prime_sum(x: int, idx: int) -> int:
        if idx >= prime_count:
            if x <= primes[-1]:
                return 0
        elif x < primes[idx]:
            return 0
        return (prime_sum_upto(x) - lower_prime_sum[idx]) % MOD

    memo: dict[int, int] = {}
    missing = -1

    def contribution(bound: int, idx: int) -> int:
        end = pi[fast_isqrt(bound)]
        total = terminal_prime_sum(bound, idx)
        if idx >= end:
            return total

        key = bound * key_base + idx
        cached = memo.get(key, missing)
        if cached is not missing:
            return cached

        for i in range(idx, end):
            p = primes[i]
            power = p
            power_limit = bound // p
            next_idx = i + 1

            while power <= power_limit:
                child_bound = bound // power
                child_end = pi[fast_isqrt(child_bound)]
                if next_idx >= child_end:
                    child = terminal_prime_sum(child_bound, next_idx)
                else:
                    child_key = child_bound * key_base + next_idx
                    child = memo.get(child_key, missing)
                    if child is missing:
                        child = contribution(child_bound, next_idx)
                total += child + p
                power *= p

        total %= MOD
        memo[key] = total
        return total

    return contribution(N, 0)


def F_small(n: int) -> int:
    lpf = [0] * (n + 1)
    for p in range(2, n + 1):
        if lpf[p] == 0:
            for j in range(p, n + 1, p):
                lpf[j] = p
    return sum(lpf[2:])


if __name__ == "__main__":
    assert F_small(10) == 32
    assert F_small(100) == 1915
    assert F_small(10000) == 10118280
    assert compute(10) == 32
    assert compute(100) == 1915
    assert compute(10000) == 10118280
    print(compute(201820182018))
