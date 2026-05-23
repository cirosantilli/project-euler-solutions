#!/usr/bin/env python
from __future__ import annotations

from fractions import Fraction

MOD: int = 1_000_000_007
TARGET_N: int = 10**18

# For even n > 3:
#   f(n) = sum(num / den * n**power) - 419
EVEN_POLYNOMIAL: tuple[tuple[int, int, int], ...] = (
    (31, 40320, 8),
    (31, 3360, 7),
    (67, 1440, 6),
    (41, 320, 5),
    (313, 1440, 4),
    (-5699, 240, 3),
    (16049, 420, 2),
    (29413, 140, 1),
)


def f_even_mod(n: int, mod: int = MOD) -> int:
    powers = [1]
    x = n % mod
    for _ in range(8):
        powers.append((powers[-1] * x) % mod)

    total = -419
    for numerator, denominator, power in EVEN_POLYNOMIAL:
        total += numerator * powers[power] * pow(denominator, mod - 2, mod)
        total %= mod
    return total


def f_even_int(n: int) -> int:
    total = Fraction(-419, 1)
    for numerator, denominator, power in EVEN_POLYNOMIAL:
        total += Fraction(numerator * n**power, denominator)
    if total.denominator != 1:
        raise ValueError("Expected integral closed-form value")
    return total.numerator


def solve() -> int:
    # Statement / derivation checkpoints.
    assert f_even_int(100) == 8658918531876
    assert f_even_mod(10000) == 377956308

    return f_even_mod(TARGET_N)


if __name__ == "__main__":
    print(solve())
