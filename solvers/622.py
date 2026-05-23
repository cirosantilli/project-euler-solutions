#!/usr/bin/env python


def factorization(number: int) -> dict[int, int]:
    factors: dict[int, int] = {}

    count = 0
    while number % 2 == 0:
        count += 1
        number //= 2
    if count:
        factors[2] = count

    factor = 3
    while factor * factor <= number:
        count = 0
        while number % factor == 0:
            count += 1
            number //= factor
        if count:
            factors[factor] = count
        factor += 2

    if number > 1:
        factors[number] = factors.get(number, 0) + 1
    return factors


def divisors_from_factorization(factors: dict[int, int]) -> list[int]:
    divisors = [1]
    for prime, exponent in factors.items():
        powers = [prime**power for power in range(exponent + 1)]
        divisors = [divisor * power for divisor in divisors for power in powers]
    return divisors


def compute(order: int) -> int:
    target_factors = factorization((1 << order) - 1)
    order_prime_divisors = tuple(factorization(order))

    total = 0
    for divisor in divisors_from_factorization(target_factors):
        if divisor == 1:
            continue
        if all(pow(2, order // prime, divisor) != 1 for prime in order_prime_divisors):
            total += divisor + 1
    return total


if __name__ == "__main__":
    assert compute(8) == 412
    print(compute(60))
