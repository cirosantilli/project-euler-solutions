#!/usr/bin/env python

PRIMES = [5, 13, 17, 29, 37, 41, 53, 61]
TARGET_EXPONENTS = [6, 3, 2, 1, 1, 1, 1, 1]


def product(values) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def B1(e: int) -> int:
    return e + 1


def B2(e: int) -> int:
    return (e + 1) ** 2


def B3(e: int) -> int:
    return ((e + 1) ** 2 + 1) // 2


def B4(e: int) -> int:
    return (e + 1) ** 3


def B5(e: int) -> int:
    return (e + 1) * (2 * e * e + 4 * e + 3) // 3


def A1(e: int) -> int:
    return (e + 1) * (e + 2) // 2


def A2(e: int) -> int:
    return (e + 1) * (e + 2) * (2 * e + 3) // 6


def A3(e: int) -> int:
    return (A2(e) + e // 2 + 1) // 2


def A4(e: int) -> int:
    value = A1(e)
    return value * value


def A5(e: int) -> int:
    return (e + 1) * (e + 2) * (e * e + 3 * e + 3) // 6


def linear_combination(exponents: list[int], factors) -> int:
    terms = [product(factor(e) for e in exponents) for factor in factors]
    return 7 * terms[0] - 14 * terms[1] - 4 * terms[2] + 8 * terms[3] + 4 * terms[4]


def exact_count_from_exponents(exponents: list[int]) -> int:
    return linear_combination(exponents, [B1, B2, B3, B4, B5])


def divisor_sum_from_exponents(exponents: list[int]) -> int:
    return linear_combination(exponents, [A1, A2, A3, A4, A5])


def odd_prime_exponents(value: int) -> list[int]:
    while value % 2 == 0:
        value //= 2

    exponents = []
    for prime in PRIMES:
        exponent = 0
        while value % prime == 0:
            value //= prime
            exponent += 1
        if exponent:
            exponents.append(exponent)

    if value != 1:
        raise ValueError("unexpected prime factor")
    return exponents


def F_of_d(d: int) -> int:
    return exact_count_from_exponents(odd_prime_exponents(d))


def compute_S_of_n() -> int:
    return divisor_sum_from_exponents(TARGET_EXPONENTS)


def _run_tests() -> None:
    assert F_of_d(1) == 1
    assert F_of_d(2) == 1
    assert F_of_d(5) == 38
    assert F_of_d(25) == 167


def main() -> None:
    _run_tests()
    print(compute_S_of_n())


if __name__ == "__main__":
    main()
