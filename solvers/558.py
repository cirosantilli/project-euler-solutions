#!/usr/bin/env python
"""
Project Euler 558 - Irrational Base

For the real root r of x^3 - x^2 - 1, the greedy representation of an
integer uses powers whose exponents are at least three apart.  The solver
precomputes enough powers of r as high-scale integers, then performs the
greedy subtraction with integer comparisons only.
"""

from __future__ import annotations

import sys
from bisect import bisect_right
from decimal import Decimal, ROUND_FLOOR, localcontext


SCALE = 10**90
DECIMAL_PRECISION = 150
MIN_EXPONENT = -200
MAX_EXPONENT = 128
DEFAULT_LIMIT = 5_000_000


def base_root() -> Decimal:
    """Return the real root of x^3 - x^2 - 1 with the active precision."""
    x = Decimal(3) / Decimal(2)
    for _ in range(24):
        x -= (x * x * x - x * x - 1) / (3 * x * x - 2 * x)
    return +x


def build_scaled_powers() -> tuple[list[int], int]:
    """
    Precompute floor(r^e * SCALE) for the target exponent window.

    The scale is intentionally much finer than the smallest retained power, so
    truncation dust cannot be mistaken for another greedy term.
    """
    with localcontext() as ctx:
        ctx.prec = DECIMAL_PRECISION
        r = base_root()
        values: dict[int, Decimal] = {0: Decimal(1), 1: r, 2: r * r}

        for exponent in range(2, MAX_EXPONENT):
            values[exponent + 1] = values[exponent] + values[exponent - 2]
        for offset in range(1, -MIN_EXPONENT + 1):
            values[-offset] = values[-offset + 3] - values[-offset + 2]

        decimal_scale = Decimal(SCALE)
        powers = [
            int(
                (values[exponent] * decimal_scale).to_integral_value(
                    rounding=ROUND_FLOOR
                )
            )
            for exponent in range(MIN_EXPONENT, MAX_EXPONENT + 1)
        ]

    return powers, -MIN_EXPONENT


def greedy_length(n: int, powers: list[int], zero_index: int) -> int:
    """Count the powers in the greedy representation of a positive integer."""
    scaled_n = n * SCALE
    lead = bisect_right(powers, scaled_n) - 1
    residual = scaled_n - powers[lead]
    terms = 1
    pos = lead - 3

    while pos >= 0:
        while pos >= 0 and powers[pos] > residual:
            pos -= 1
        if pos < 0:
            break
        residual -= powers[pos]
        terms += 1
        pos -= 3

    return terms


def S(
    limit: int, powers: list[int] | None = None, zero_index: int | None = None
) -> int:
    """
    Compute sum_{n=1..limit} l(n^2).

    Since n^2 is increasing, the leading exponent pointer only moves forward.
    """
    if powers is None or zero_index is None:
        powers, zero_index = build_scaled_powers()

    lead = zero_index
    square = 0
    odd_increment = SCALE
    two = 2 * SCALE
    total = 0
    power_count = len(powers)

    for _ in range(limit):
        square += odd_increment
        odd_increment += two

        while lead + 1 < power_count and powers[lead + 1] <= square:
            lead += 1

        residual = square - powers[lead]
        terms = 1
        pos = lead - 3

        while pos >= 0:
            while pos >= 0 and powers[pos] > residual:
                pos -= 1
            if pos < 0:
                break
            residual -= powers[pos]
            terms += 1
            pos -= 3

        total += terms

    return total


def solve(limit: int = DEFAULT_LIMIT) -> int:
    powers, zero_index = build_scaled_powers()
    return S(limit, powers, zero_index)


def _self_test() -> None:
    powers, zero_index = build_scaled_powers()
    assert greedy_length(3, powers, zero_index) == 4
    assert greedy_length(4, powers, zero_index) == 4
    assert greedy_length(10, powers, zero_index) == 3
    assert S(10, powers, zero_index) == 61
    assert S(1000, powers, zero_index) == 19403


if __name__ == "__main__":
    _self_test()
    n = DEFAULT_LIMIT
    if len(sys.argv) >= 2:
        n = int(sys.argv[1])
    print(solve(n))
