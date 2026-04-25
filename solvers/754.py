#!/usr/bin/env python
"""
Project Euler 754: Product of Gauss Factorials

Compute the product of Gauss factorials GF(n), 1 <= n <= 10^8, modulo
1_000_000_007.
"""

from math import gcd


LIMIT = 100_000_000
MOD = 1_000_000_007
EXP_MOD = MOD - 1


def gauss_factorial(n: int) -> int:
    result = 1
    for k in range(1, n):
        if gcd(k, n) == 1:
            result *= k
    return result


def product_gauss_factorials(n: int) -> int:
    result = 1
    for k in range(1, n + 1):
        result *= gauss_factorial(k)
    return result


def mobius_sieve(n: int) -> bytearray:
    # Encode mu as 0 -> 0, 1 -> +1, 2 -> -1.
    mu = bytearray(n + 1)
    mu[1] = 1
    composite = bytearray(n + 1)
    primes: list[int] = []

    for x in range(2, n + 1):
        if not composite[x]:
            primes.append(x)
            mu[x] = 2

        mux = mu[x]
        for p in primes:
            y = x * p
            if y > n:
                break
            composite[y] = 1
            if x % p == 0:
                break
            if mux == 1:
                mu[y] = 2
            elif mux == 2:
                mu[y] = 1

    return mu


def quotient_ranges(n: int) -> list[tuple[int, int, int]]:
    ranges = []
    lo = 1
    while lo <= n:
        q = n // lo
        hi = n // q
        ranges.append((lo, hi, q))
        lo = hi + 1
    return ranges


def superfactorials_at(keys: list[int]) -> dict[int, int]:
    values: dict[int, int] = {}
    keys = sorted(set(keys))
    pos = 0
    factorial = 1
    superfactorial = 1

    if keys and keys[0] == 0:
        values[0] = 1
        pos = 1

    stop = keys[-1] if keys else 0
    for x in range(1, stop + 1):
        factorial = factorial * x % MOD
        superfactorial = superfactorial * factorial % MOD
        while pos < len(keys) and keys[pos] == x:
            values[x] = superfactorial
            pos += 1
    return values


def solve(limit: int = LIMIT) -> int:
    mu = mobius_sieve(limit)
    ranges = quotient_ranges(limit)
    superfactorial = superfactorials_at([q - 1 for _, _, q in ranges])

    result = 1
    for lo, hi, q in ranges:
        exponent = q * (q - 1) // 2 % EXP_MOD
        pos_product = 1
        neg_product = 1
        mu_sum = 0

        for d in range(lo, hi + 1):
            sign = mu[d]
            if sign == 1:
                pos_product = pos_product * d % MOD
                mu_sum += 1
            elif sign == 2:
                neg_product = neg_product * d % MOD
                mu_sum -= 1

        if exponent:
            result = result * pow(pos_product, exponent, MOD) % MOD
            result = result * pow(neg_product, (-exponent) % EXP_MOD, MOD) % MOD

        sf_power = mu_sum % EXP_MOD
        if sf_power:
            result = result * pow(superfactorial[q - 1], sf_power, MOD) % MOD

    return result


def main() -> None:
    assert gauss_factorial(10) == 189
    assert product_gauss_factorials(10) == 23044331520000
    assert solve(10) == 331358692

    print(solve())


if __name__ == "__main__":
    main()
