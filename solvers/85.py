#!/usr/bin/env python
from math import isqrt


def rectangle_count(m: int, n: int) -> int:
    return (m * (m + 1) * n * (n + 1)) // 4


def best_grid_area_near(target: int) -> tuple[int, int, int]:
    """
    Return the closest positive grid, choosing the least (m, n) on ties, with m <= n.
    """
    best_m, best_n = 1, 1
    best_diff = abs(1 - target)
    m = 1
    while True:
        vertical = m * (m + 1) // 2
        if vertical * vertical > target + best_diff:
            return best_m, best_n, best_m * best_n
        quotient = max(0, target // vertical)
        below = max(m, (isqrt(1 + 8 * quotient) - 1) // 2)
        for n in (below, below + 1):
            diff = abs(rectangle_count(m, n) - target)
            if diff < best_diff:
                best_diff = diff
                best_m, best_n = m, n
        m += 1


def solve(target: int) -> int:
    """
    @formalsw: Py from project_euler_statements.p85 import naive
    @formalsw: Returns
    @formalsw: Equiv(naive)
    """
    return best_grid_area_near(target)[2]


if __name__ == "__main__":
    assert rectangle_count(3, 2) == 18
    print(solve(2_000_000))
