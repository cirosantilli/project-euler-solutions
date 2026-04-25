#!/usr/bin/env python
"""
Project Euler 259: Reachable Numbers

Use the digits 1..9 in order, allowing concatenation, binary arithmetic
operations, and arbitrary parentheses.  Sum the distinct positive integers that
can be reached exactly.
"""

from functools import cache
from math import gcd


DIGITS = "123456789"


def normalize(num: int, den: int) -> tuple[int, int]:
    if den < 0:
        num = -num
        den = -den
    g = gcd(num, den)
    return num // g, den // g


def add_values(
    out: set[tuple[int, int]], left: tuple[int, int], right: tuple[int, int]
) -> None:
    an, ad = left
    bn, bd = right

    out.add(normalize(an * bd + bn * ad, ad * bd))
    out.add(normalize(an * bd - bn * ad, ad * bd))
    out.add(normalize(an * bn, ad * bd))
    if bn:
        out.add(normalize(an * bd, ad * bn))


def reachable_values(digits: str) -> set[tuple[int, int]]:
    @cache
    def interval(lo: int, hi: int) -> frozenset[tuple[int, int]]:
        values: set[tuple[int, int]] = {(int(digits[lo:hi]), 1)}

        for split in range(lo + 1, hi):
            left_values = interval(lo, split)
            right_values = interval(split, hi)
            for left in left_values:
                for right in right_values:
                    add_values(values, left, right)

        return frozenset(values)

    return set(interval(0, len(digits)))


def solve(last_digit: int = 9) -> int:
    values = reachable_values(DIGITS[:last_digit])
    integers = {num for num, den in values if den == 1 and num > 0}
    return sum(integers)


def main() -> None:
    values = reachable_values(DIGITS)
    assert (42, 1) in values

    print(solve())


if __name__ == "__main__":
    main()
