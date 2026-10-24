#!/usr/bin/env python
from __future__ import annotations

import math


def triangle(n: int) -> int:
    return n * (n + 1) // 2


def pentagonal(n: int) -> int:
    return n * (3 * n - 1) // 2


def hexagonal(n: int) -> int:
    return n * (2 * n - 1)


def is_pentagonal(x: int) -> bool:
    # x = n(3n-1)/2  =>  3n^2 - n - 2x = 0
    # n = (1 + sqrt(1 + 24x)) / 6 must be a positive integer
    d = 1 + 24 * x
    s = math.isqrt(d)
    if s * s != d:
        return False
    return (1 + s) % 6 == 0


def solve() -> int:
    """
    @formalsw: Py from project_euler_statements.p45 import naive_default
    @formalsw: Equiv(naive_default)
    """
    # Every hexagonal number is triangular, so we only need H_n that is pentagonal.
    n = 144  # next after H_143
    while True:
        h = hexagonal(n)
        if is_pentagonal(h):
            return h
        n += 1


if __name__ == "__main__":
    # Given check from statement:
    assert triangle(285) == 40755
    assert pentagonal(165) == 40755
    assert hexagonal(143) == 40755
    assert triangle(285) == pentagonal(165) == hexagonal(143)

    print(solve())
