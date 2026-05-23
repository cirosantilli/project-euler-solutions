#!/usr/bin/env python

BASE = 14
BASE_DIGITS = "0123456789abcd"
INVERSES = {1: 1, 3: 5, 5: 3, 9: 11, 11: 9, 13: 13}


def to_base_14(value: int) -> str:
    if value == 0:
        return "0"

    digits = []
    while value:
        value, digit = divmod(value, BASE)
        digits.append(BASE_DIGITS[digit])
    return "".join(reversed(digits))


def initial_state(root: int) -> tuple[int, int, int, int, int]:
    modulus = BASE
    quotient = (root * root - root) // modulus
    digit_sum = root
    leading_digit = root
    return root, quotient, modulus, digit_sum, leading_digit


def lift(state: tuple[int, int, int, int, int]) -> tuple[int, int, int, int, int]:
    value, quotient, modulus, digit_sum, _ = state
    coefficient = (2 * value - 1) % BASE
    leading_digit = (-quotient * INVERSES[coefficient]) % BASE

    next_value = value + leading_digit * modulus
    next_quotient = (
        quotient
        + (2 * value - 1) * leading_digit
        + leading_digit * leading_digit * modulus
    ) // BASE

    return (
        next_value,
        next_quotient,
        modulus * BASE,
        digit_sum + leading_digit,
        leading_digit,
    )


def solve(max_digits: int) -> str:
    branches = [initial_state(7), initial_state(8)]
    total = 1

    for length in range(1, max_digits + 1):
        for _, _, _, digit_sum, leading_digit in branches:
            if leading_digit:
                total += digit_sum
        if length != max_digits:
            branches = [lift(branch) for branch in branches]

    return to_base_14(total)


if __name__ == "__main__":
    assert solve(9) == "2d8"
    print(solve(10000))
