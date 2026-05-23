#!/usr/bin/env python
"""
Project Euler 752 - Powers of 1 + sqrt(7)

For gcd(n, 6) = 1, g(n) is the multiplicative order of 1 + sqrt(7) in
(Z/nZ)[sqrt(7)].  The order is computed on prime powers, then combined for
composite n by lcm via the Chinese remainder theorem.
"""

from array import array
from math import gcd


def smallest_prime_factors(limit: int) -> tuple[array, list[int]]:
    spf = array("I", [0]) * (limit + 1)
    primes: list[int] = []

    for i in range(2, limit + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)

        for p in primes:
            value = i * p
            if value > limit or p > spf[i]:
                break
            spf[value] = p

    return spf, primes


def unique_prime_factors(n: int, spf: array) -> list[int]:
    factors: list[int] = []
    while n > 1:
        p = spf[n]
        factors.append(p)
        while n % p == 0:
            n //= p
    return factors


def quadratic_power(exponent: int, modulus: int) -> tuple[int, int]:
    """Return (1 + sqrt(7))**exponent as (a, b) modulo modulus."""
    result_a, result_b = 1, 0
    base_a, base_b = 1, 1

    while exponent:
        if exponent & 1:
            result_a, result_b = (
                (base_a * result_a + 7 * base_b * result_b) % modulus,
                (base_a * result_b + base_b * result_a) % modulus,
            )

        base_a, base_b = (
            (base_a * base_a + 7 * base_b * base_b) % modulus,
            (2 * base_a * base_b) % modulus,
        )
        exponent >>= 1

    return result_a, result_b


def prime_order(p: int, spf: array) -> int:
    if p == 7:
        return 7

    if pow(7, (p - 1) // 2, p) == 1:
        order = p - 1
        factors = unique_prime_factors(p - 1, spf)
    else:
        order = (p - 1) * (p + 1)
        factors = unique_prime_factors(p - 1, spf)
        factors += unique_prime_factors(p + 1, spf)

    for q in dict.fromkeys(factors):
        while order % q == 0 and quadratic_power(order // q, p) == (1, 0):
            order //= q

    return order


def build_prime_power_orders(limit: int, spf: array, primes: list[int]) -> array:
    orders = array("Q", [0]) * (limit + 1)

    for p in primes:
        if p > limit:
            break
        if p < 5:
            continue

        order = prime_order(p, spf)
        prime_power = p
        lifted_order = order
        orders[prime_power] = lifted_order

        while prime_power * p <= limit:
            prime_power *= p
            if quadratic_power(lifted_order, prime_power) != (1, 0):
                lifted_order *= p
            orders[prime_power] = lifted_order

    return orders


def G(limit: int) -> int:
    spf, primes = smallest_prime_factors(limit + 1)
    orders = build_prime_power_orders(limit, spf, primes)

    values = array("Q", [0]) * (limit + 1)
    total = 0

    for n in range(2, limit + 1):
        if n % 2 == 0 or n % 3 == 0:
            continue

        p = spf[n]
        remaining = n
        prime_power = 1
        while remaining % p == 0:
            remaining //= p
            prime_power *= p

        previous = values[remaining] if remaining > 1 else 1
        component_order = orders[prime_power]
        value = previous // gcd(previous, component_order) * component_order
        values[n] = value
        total += value

    return total


def g(n: int) -> int:
    if n % 2 == 0 or n % 3 == 0:
        return 0

    spf, primes = smallest_prime_factors(n + 1)
    orders = build_prime_power_orders(n, spf, primes)

    value = 1
    remaining = n
    while remaining > 1:
        p = spf[remaining]
        prime_power = 1
        while remaining % p == 0:
            remaining //= p
            prime_power *= p

        component_order = orders[prime_power]
        value = value // gcd(value, component_order) * component_order

    return value


if __name__ == "__main__":
    assert g(3) == 0
    assert g(5) == 12
    assert G(10**2) == 28891
    assert G(10**3) == 13131583
    print(G(10**6))
