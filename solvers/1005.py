#!/usr/bin/env python
from __future__ import annotations

from math import isqrt


MODULUS = 1_000_000_000


def primes_up_to(limit: int) -> list[int]:
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[:2] = b"\x00\x00"
    for prime in range(2, isqrt(limit) + 1):
        if sieve[prime]:
            start = prime * prime
            sieve[start : limit + 1 : prime] = b"\x00" * (
                (limit - start) // prime + 1
            )
    return [value for value in range(limit + 1) if sieve[value]]


def median_prime_list(total: int) -> list[int]:
    primes = primes_up_to(total)
    count = [[0] * (total + 1) for _ in range(len(primes) + 1)]
    count[-1][0] = 1

    for index in range(len(primes) - 1, -1, -1):
        prime = primes[index]
        row = count[index]
        next_row = count[index + 1]
        row[:] = next_row[:]
        for subtotal in range(prime, total + 1):
            row[subtotal] += next_row[subtotal - prime]

    assert count[0][total] > 0
    rank = (count[0][total] - 1) // 2
    result: list[int] = []
    subtotal = total
    start_index = 0

    while subtotal:
        for index in range(start_index, len(primes)):
            prime = primes[index]
            if prime > subtotal:
                break
            block_size = count[index + 1][subtotal - prime]
            if rank >= block_size:
                rank -= block_size
            else:
                result.append(prime)
                subtotal -= prime
                start_index = index + 1
                break
    return result


def prime_product_last_nine_digits(total: int) -> int:
    product = 1
    for prime in median_prime_list(total):
        product = product * prime % MODULUS
    return product


def main() -> None:
    assert median_prime_list(20) == [2, 7, 11]
    print(prime_product_last_nine_digits(2026))


if __name__ == "__main__":
    main()
