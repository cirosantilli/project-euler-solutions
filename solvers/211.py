#!/usr/bin/env python
"""
Project Euler 211: Divisor Square Sum

Generate each prime factorization below the limit once, carrying sigma_2(n)
multiplicatively:

    sigma_2(p^e) = 1 + p^2 + ... + p^(2e)
    sigma_2(p^(e+1)) = p^2 * sigma_2(p^e) + 1

This avoids summing divisor squares for every integer or every divisor.
"""

from math import isqrt


def prime_list_upto(limit: int) -> list[int]:
    if limit < 2:
        return []

    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[0] = sieve[1] = 0
    root = isqrt(limit)
    for p in range(2, root + 1):
        if sieve[p]:
            start = p * p
            sieve[start : limit + 1 : p] = b"\x00" * (((limit - start) // p) + 1)

    return [p for p in range(2, limit + 1) if sieve[p]]


def is_square(n: int) -> bool:
    root = isqrt(n)
    return root * root == n


def solve(limit: int) -> int:
    max_n = limit - 1
    primes = prime_list_upto(max_n)
    total = 1  # sigma_2(1) = 1

    def visit(first_prime_index: int, current_sigma: int, current_n: int) -> int:
        subtotal = 0
        for index in range(first_prime_index, len(primes)):
            p = primes[index]
            if current_n > max_n // p:
                break

            prime_square = p * p
            prime_power = p
            sigma_factor = 1

            while current_n <= max_n // prime_power:
                candidate_n = current_n * prime_power
                sigma_factor = sigma_factor * prime_square + 1
                candidate_sigma = current_sigma * sigma_factor

                if is_square(candidate_sigma):
                    subtotal += candidate_n

                subtotal += visit(index + 1, candidate_sigma, candidate_n)

                if prime_power > max_n // p:
                    break
                prime_power *= p

        return subtotal

    total += visit(0, 1, 1)
    return total


if __name__ == "__main__":
    print(solve(64_000_000))
