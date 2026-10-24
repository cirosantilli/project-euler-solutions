#!/usr/bin/env python
from __future__ import annotations

from math import gcd


def best_fraction_left_of(target_n: int, target_d: int, max_d: int) -> tuple[int, int]:
    """
    Returns (n, d) = reduced proper fraction with d <= max_d, n/d < target_n/target_d,
    and n/d as large as possible.

    Requires 0 < target_n < target_d. Return (0, 1) if there is no positive candidate.
    """
    best_n, best_d = 0, 1

    for d in range(1, max_d + 1):
        # Largest n such that n/d < target_n/target_d  =>  n < target_n*d/target_d
        n = (target_n * d - 1) // target_d
        if n <= 0:
            continue

        if gcd(n, d) != 1:
            continue

        # Compare n/d > best_n/best_d via cross-multiplication
        if n * best_d > best_n * d:
            best_n, best_d = n, d

    return best_n, best_d


def solve(limit: int, num: int = 3, den: int = 7) -> int:
    """
    @formalsw: Py from project_euler_statements.p71 import naive
    @formalsw: Requires(0 < num)
    @formalsw: Requires(num < den)
    @formalsw: Returns
    @formalsw: Equiv(naive)
    """
    return best_fraction_left_of(num, den, limit)[0]


if __name__ == "__main__":
    # Test from the statement: for d <= 8, fraction immediately left of 3/7 is 2/5
    assert best_fraction_left_of(3, 7, 8) == (2, 5)

    print(solve(1_000_000))
