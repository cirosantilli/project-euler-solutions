#!/usr/bin/env python

from array import array
from math import isqrt


def prime_flags(limit):
    half = (limit >> 1) + 1
    flags = bytearray(b"\x01") * half
    flags[0] = 0

    for p in range(3, isqrt(limit) + 1, 2):
        if flags[p >> 1]:
            start = (p * p) >> 1
            count = ((half - 1 - start) // p) + 1
            flags[start::p] = b"\x00" * count

    return flags


def iter_primes(flags, limit):
    if limit >= 2:
        yield 2
    for i in range(1, (limit >> 1) + 1):
        if flags[i]:
            yield (i << 1) + 1


def factorial_valuation(n, p):
    total = 0
    while n:
        n //= p
        total += n
    return total


def prime_power_threshold(p, exponent):
    lo = 1
    hi = p * exponent
    while lo < hi:
        mid = (lo + hi) >> 1
        if factorial_valuation(mid, p) >= exponent:
            hi = mid
        else:
            lo = mid + 1
    return lo


def solve(limit=100_000_000):
    flags = prime_flags(limit)
    least = array("I", [0]) * (limit + 1)

    values = least
    for p in iter_primes(flags, limit):
        for multiple in range(p, limit + 1, p):
            values[multiple] = p

    for p in iter_primes(flags, isqrt(limit)):
        power = p * p
        exponent = 2
        while power <= limit:
            threshold = prime_power_threshold(p, exponent)
            for multiple in range(power, limit + 1, power):
                if values[multiple] < threshold:
                    values[multiple] = threshold
            power *= p
            exponent += 1

    return sum(values[2:])


def naive(n):
    factorial = 1
    result = 0
    while factorial % n:
        result += 1
        factorial = (factorial * result) % n
    return result


def main():
    assert naive(10) == 5
    assert naive(25) == 10
    assert solve(100) == 2012
    print(solve())


if __name__ == "__main__":
    main()
