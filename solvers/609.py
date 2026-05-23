#!/usr/bin/env python

from array import array
import math


MOD = 1_000_000_007


def prime_sieve(limit: int) -> bytearray:
    is_prime = bytearray(b"\x01") * (limit + 1)
    if limit >= 0:
        is_prime[0] = 0
    if limit >= 1:
        is_prime[1] = 0

    for p in range(2, math.isqrt(limit) + 1):
        if is_prime[p]:
            start = p * p
            is_prime[start : limit + 1 : p] = b"\x00" * (((limit - start) // p) + 1)

    return is_prime


def prime_count_table(is_prime: bytearray, limit: int) -> array:
    pi = array("I", [0]) * (limit + 1)
    count = 0
    for x in range(1, limit + 1):
        if is_prime[x]:
            count += 1
        pi[x] = count
    return pi


def add_pending(
    pending: dict[int, dict[int, int]], value: int, nonprime_count: int, weight: int
) -> None:
    counts = pending.get(value)
    if counts is None:
        pending[value] = {nonprime_count: weight}
    else:
        counts[nonprime_count] = counts.get(nonprime_count, 0) + weight


def P(limit: int, mod: int | None = MOD) -> int:
    is_prime = prime_sieve(limit)
    primes = [2] if limit >= 2 else []
    primes.extend(x for x in range(3, limit + 1, 2) if is_prime[x])

    m = len(primes)
    pi = prime_count_table(is_prime, m)
    is_nonprime = bytearray(m + 1)
    for x in range(1, m + 1):
        is_nonprime[x] = 1 - is_prime[x]

    buckets = [0] * 64
    pending: dict[int, dict[int, int]] = {}

    for first_value in range(m, 0, -1):
        tail_counts = pending.pop(first_value, None)
        if first_value < m:
            composite_starts = primes[first_value] - primes[first_value - 1] - 1
        else:
            composite_starts = limit - primes[-1]

        nonprime_delta = is_nonprime[first_value]
        next_value = pi[first_value]

        prime_bucket = nonprime_delta
        buckets[prime_bucket] += 1
        if next_value:
            add_pending(pending, next_value, prime_bucket, 1)

        composite_bucket = nonprime_delta + 1
        buckets[composite_bucket] += composite_starts
        if composite_starts and next_value:
            add_pending(pending, next_value, composite_bucket, composite_starts)

        if tail_counts is None:
            continue

        for nonprime_count, weight in tail_counts.items():
            bucket = nonprime_count + nonprime_delta
            buckets[bucket] += weight
            if next_value:
                add_pending(pending, next_value, bucket, weight)

    total = 1
    for count in buckets:
        if count:
            total *= count
            if mod is not None:
                total %= mod

    return total if mod is None else total % mod


if __name__ == "__main__":
    assert P(10, mod=None) == 648
    assert P(100, mod=None) == 31038676032
    print(P(10**8))
