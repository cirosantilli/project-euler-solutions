#!/usr/bin/env python

import math


def primes_upto(limit: int) -> list[int]:
    sieve = bytearray(b"\x01") * (limit + 1)
    if limit >= 0:
        sieve[0] = 0
    if limit >= 1:
        sieve[1] = 0
    for number in range(2, math.isqrt(limit) + 1):
        if sieve[number]:
            start = number * number
            sieve[start : limit + 1 : number] = b"\x00" * (
                (limit - start) // number + 1
            )
    return [number for number in range(2, limit + 1) if sieve[number]]


def is_prime(value: int) -> bool:
    if value < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29)
    for prime in small_primes:
        if value == prime:
            return True
        if value % prime == 0:
            return False

    d = value - 1
    shifts = 0
    while d % 2 == 0:
        d //= 2
        shifts += 1

    if value < 4_759_123_141:
        bases = (2, 7, 61)
    elif value < 1_122_004_669_633:
        bases = (2, 13, 23, 1_662_803)
    else:
        bases = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)

    for base in bases:
        x = pow(base, d, value)
        if x == 1 or x == value - 1:
            continue
        for _ in range(shifts - 1):
            x = (x * x) % value
            if x == value - 1:
                break
        else:
            return False

    return True


def is_prime_proof(value: int) -> bool:
    digits = []
    powers = []
    power = 1
    remaining = value
    while remaining:
        digits.append(remaining % 10)
        powers.append(power)
        remaining //= 10
        power *= 10

    leading_position = len(digits) - 1
    for position, original_digit in enumerate(digits):
        first_digit = 1 if position == leading_position else 0
        place = powers[position]
        for digit in range(first_digit, 10):
            if digit == original_digit:
                continue
            if position == 0 and (digit % 2 == 0 or digit == 5):
                continue
            modified = value + (digit - original_digit) * place
            if is_prime(modified):
                return False
    return True


def generate_squbes(limit: int) -> list[int]:
    primes = primes_upto(math.isqrt(limit // 8) + 1)
    squbes = []

    for p in primes:
        p_squared = p * p
        if p_squared * 8 > limit:
            break
        for q in primes:
            if q == p:
                continue
            value = p_squared * q * q * q
            if value > limit:
                break
            squbes.append(value)

    squbes.sort()
    return squbes


def solve(target: int, token: str) -> tuple[int, int]:
    limit = 1_000_000_000_000

    while True:
        count = 0
        second_value = -1

        for value in generate_squbes(limit):
            if token not in str(value) or not is_prime_proof(value):
                continue

            count += 1
            if count == 2:
                second_value = value
            if count == target:
                return value, second_value

        limit *= 2


if __name__ == "__main__":
    value, second_value = solve(200, "200")
    assert second_value == 1992008
    print(value)
