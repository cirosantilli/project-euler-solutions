#!/usr/bin/env python
from __future__ import annotations

from math import isqrt


def digit_sum(n: int) -> int:
    return sum(int(c) for c in str(n))


def sqrt_digits_sum(n: int, digits: int) -> int:
    root = isqrt(n)
    if root * root == n:
        return 0
    width = len(str(root))
    if digits < width:
        prefix = root // 10 ** (width - digits)
    else:
        prefix = isqrt(n * 10 ** (2 * (digits - width)))
    return digit_sum(prefix)


def solve(limit: int = 100, digits: int = 100) -> int:
    """
    @formalsw: Requires(limit >= 0)
    @formalsw: Requires(digits >= 1)
    @formalsw: Py from project_euler_statements.p80 import naive
    @formalsw: Returns
    @formalsw: Equiv(naive)
    """
    total = 0
    for n in range(1, limit + 1):
        total += sqrt_digits_sum(n, digits)
    return total


if __name__ == "__main__":
    assert sqrt_digits_sum(2, 100) == 475
    print(solve())
