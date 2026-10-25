#!/usr/bin/env python
from __future__ import annotations

from math import gcd


def count_between(d: int, lower_num: int, lower_den: int, upper_num: int, upper_den: int) -> int:
    """
    Count positive reduced proper fractions at denominator d inside the open interval.

    @formalsw: Requires(lower_den > 0)
    @formalsw: Requires(upper_den > 0)
    @formalsw: Returns
    """
    n_start = max(1, lower_num * d // lower_den + 1)
    n_end = min(d - 1, (upper_num * d - 1) // upper_den)
    if n_start > n_end:
        return 0
    total = 0
    for n in range(n_start, n_end + 1):
        if gcd(n, d) == 1:
            total += 1
    return total


def solve(
    limit: int,
    lower_num: int = 1,
    lower_den: int = 3,
    upper_num: int = 1,
    upper_den: int = 2,
) -> int:
    """
    Count positive reduced proper fractions inside the supplied open rational interval.

    @formalsw: Py from project_euler_statements.p73 import naive
    @formalsw: Requires(lower_den > 0)
    @formalsw: Requires(upper_den > 0)
    @formalsw: Returns
    @formalsw: Equiv(naive)
    """
    total = 0
    for d in range(2, limit + 1):
        total += count_between(d, lower_num, lower_den, upper_num, upper_den)
    return total


if __name__ == "__main__":
    assert solve(8) == 3
    print(solve(12_000))
