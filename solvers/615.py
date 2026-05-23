#!/usr/bin/env python

import math


MOD = 123454321
TARGET_RANK = 1_000_000
TARGET_OMEGA = 1_000_000


def primes_upto(limit: int) -> list[int]:
    is_prime = bytearray(b"\x01") * (limit + 1)
    is_prime[0:2] = b"\x00\x00"
    for p in range(2, math.isqrt(limit) + 1):
        if is_prime[p]:
            start = p * p
            is_prime[start : limit + 1 : p] = b"\x00" * (
                ((limit - start) // p) + 1
            )
    return [2] + [p for p in range(3, limit + 1, 2) if is_prime[p]]


PRIMES = primes_upto(300_000)
PRIME_LOGS = [math.log(p) for p in PRIMES]
LOG3 = math.log(3)


def values_with_enough_prime_factors(threshold: int, collect: bool) -> tuple[int, list[int]]:
    limit = 3**threshold
    log_limit = threshold * LOG3
    values: list[int] = []
    count = 0

    def search(start_index: int, product: int, used: int, log_product: float) -> None:
        nonlocal count

        if used >= threshold:
            count += 1
            if collect:
                values.append(product)

        remaining = threshold - used
        for index in range(start_index, len(PRIMES)):
            prime = PRIMES[index]
            next_product = product * prime
            if next_product > limit:
                break
            if (
                remaining > 0
                and log_product + remaining * PRIME_LOGS[index] > log_limit + 1e-12
            ):
                break
            search(index, next_product, used + 1, log_product + PRIME_LOGS[index])

    search(0, 1, 0, 0.0)
    return count, values


def kth_value_at_working_threshold(rank: int) -> tuple[int, int]:
    threshold = 1
    while True:
        threshold += 1
        count, _ = values_with_enough_prime_factors(threshold, False)
        if count >= rank:
            _, values = values_with_enough_prime_factors(threshold, True)
            values.sort()
            return threshold, values[rank - 1]


def solve() -> int:
    threshold, value = kth_value_at_working_threshold(TARGET_RANK)
    return value * pow(2, TARGET_OMEGA - threshold, MOD) % MOD


if __name__ == "__main__":
    # The fifth number with at least five prime factors is 80.
    small_threshold, small_value = kth_value_at_working_threshold(5)
    assert small_value * (2 ** (5 - small_threshold)) == 80
    print(solve())
