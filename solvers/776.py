#!/usr/bin/env python3
"""
Project Euler 776: Digit Sum Division

For n >= 1, let d(n) be the sum of decimal digits of n.
Define F(N) = sum_{n=1..N} n / d(n).

This script computes F(1234567890123456789) and prints the result in scientific
notation with 12 digits after the decimal point (lowercase 'e').

No external libraries are used (only Python standard library).
"""

from __future__ import annotations

from decimal import Decimal, getcontext, ROUND_HALF_UP
from fractions import Fraction
from typing import List


def sum_by_digit_sum_upto(n: int) -> List[int]:
    """
    Return an array S where S[s] = sum of all integers x in [0, n] such that
    digit_sum(x) == s.

    Uses a digit DP over the decimal representation of n, allowing leading zeros
    so that all shorter numbers are included naturally.
    """
    if n < 0:
        raise ValueError("n must be non-negative")

    digits = list(map(int, str(n)))
    L = len(digits)
    max_sum = 9 * L

    # dp_*_tight: prefixes equal to n so far; dp_*_loose: already smaller.
    cnt_tight = [0] * (max_sum + 1)
    sum_tight = [0] * (max_sum + 1)
    cnt_tight[0] = 1

    cnt_loose = [0] * (max_sum + 1)
    sum_loose = [0] * (max_sum + 1)

    for lim in digits:
        ncnt_tight = [0] * (max_sum + 1)
        nsum_tight = [0] * (max_sum + 1)
        ncnt_loose = [0] * (max_sum + 1)
        nsum_loose = [0] * (max_sum + 1)

        # Transition from loose states (next digit can be 0..9).
        for s, c in enumerate(cnt_loose):
            if not c:
                continue
            v10 = sum_loose[s] * 10
            for d in range(10):
                ns = s + d
                ncnt_loose[ns] += c
                nsum_loose[ns] += v10 + c * d

        # Transition from tight states (next digit restricted by lim).
        for s, c in enumerate(cnt_tight):
            if not c:
                continue
            v10 = sum_tight[s] * 10
            for d in range(lim + 1):
                ns = s + d
                if d == lim:
                    ncnt_tight[ns] += c
                    nsum_tight[ns] += v10 + c * d
                else:
                    ncnt_loose[ns] += c
                    nsum_loose[ns] += v10 + c * d

        cnt_tight, sum_tight = ncnt_tight, nsum_tight
        cnt_loose, sum_loose = ncnt_loose, nsum_loose

    return [sum_tight[s] + sum_loose[s] for s in range(max_sum + 1)]


def F_fraction(n: int) -> Fraction:
    """
    Exact value of F(n) as a Fraction. Intended for small n (tests).
    """
    sums = sum_by_digit_sum_upto(n)
    total = Fraction(0, 1)
    for s in range(1, len(sums)):
        if sums[s]:
            total += Fraction(sums[s], s)
    return total


def F_decimal(n: int, prec: int = 100) -> Decimal:
    """
    High-precision decimal evaluation of F(n).
    """
    getcontext().prec = prec
    getcontext().rounding = ROUND_HALF_UP

    sums = sum_by_digit_sum_upto(n)
    total = Decimal(0)
    for s in range(1, len(sums)):
        if sums[s]:
            total += Decimal(sums[s]) / Decimal(s)
    return total


def format_scientific_12(x: Decimal) -> str:
    """
    Format Decimal x as mantissa with 12 digits after the decimal point and a
    lowercase 'e', with no '+' sign in the exponent.
    Example: 1.187764610390e3
    """
    s = format(x, ".12E")  # e.g. '1.234567890123E+45'
    mant, exp = s.split("E")
    return f"{mant}e{int(exp)}"


def _self_test() -> None:
    # Test values from the problem statement.
    assert F_fraction(10) == Fraction(19, 1)
    assert format_scientific_12(F_decimal(123)) == "1.187764610390e3"
    assert format_scientific_12(F_decimal(12345)) == "4.855801996238e6"


def main() -> None:
    _self_test()
    n = 1234567890123456789
    ans = F_decimal(n, prec=120)
    print(format_scientific_12(ans))


if __name__ == "__main__":
    main()
