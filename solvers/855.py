#!/usr/bin/env python3
"""Project Euler 855: Delphi Paper

Closed form:
    S(a,b) = (a!)^b * (b!)^a / ((ab)!)^2

The program computes this value exactly as a Fraction and prints it in scientific
notation with 10 digits after the decimal point, as requested by the problem.

No third-party libraries are used.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP, getcontext
from fractions import Fraction
from math import factorial
from typing import Tuple


def s_fraction(a: int, b: int) -> Fraction:
    """Return S(a,b) exactly as a reduced Fraction."""
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be positive integers")

    num = factorial(a) ** b * factorial(b) ** a
    den = factorial(a * b) ** 2
    return Fraction(num, den)


def _format_scientific(frac: Fraction, digits_after_decimal: int = 10) -> str:
    """Format a positive Fraction as m.e, with exactly `digits_after_decimal` digits
    after the decimal point in the mantissa and a lowercase 'e'.

    Example (digits_after_decimal=10): 1.2345678901e-123
    """
    if frac <= 0:
        raise ValueError("Expected a positive value")

    # Enough precision to round the mantissa correctly.
    getcontext().prec = digits_after_decimal + 50

    x = Decimal(frac.numerator) / Decimal(frac.denominator)

    # Decimal.adjusted(): exponent of the most significant digit.
    e = x.adjusted()
    m = x.scaleb(-e)  # now 1 <= m < 10

    q = Decimal("1." + "0" * digits_after_decimal)
    m = m.quantize(q, rounding=ROUND_HALF_UP)

    # Handle carry: e.g. 9.999... -> 10.000...
    if m == Decimal(10):
        m = Decimal(1).quantize(q)  # 1.0000...
        e += 1

    return f"{m:.{digits_after_decimal}f}e{e}"


def solve(a: int = 5, b: int = 8) -> str:
    """Compute the required answer string for the given (a,b)."""
    return _format_scientific(s_fraction(a, b), digits_after_decimal=10)


def _self_test() -> None:
    """Asserts for the example values stated in the problem."""
    assert s_fraction(2, 2) == Fraction(1, 36)
    assert s_fraction(2, 3) == Fraction(1, 1800)

    # A couple of sanity checks (not from the statement, but cheap and helpful).
    assert s_fraction(1, 1) == Fraction(1, 1)
    assert s_fraction(1, 5) == Fraction(1, factorial(5))


def main() -> None:
    _self_test()
    print(solve())


if __name__ == "__main__":
    main()
