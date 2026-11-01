#!/usr/bin/env python
from math import isqrt


def count_new_solutions(c: int) -> int:
    """
    Count cuboids whose largest side is c by grouping the other sides by their sum.
    """
    total = 0
    c2 = c * c
    for s in range(2, 2 * c + 1):
        d = s * s + c2
        t = isqrt(d)
        if t * t == d:
            lo = max(1, s - c)
            hi = s // 2
            total += hi - lo + 1
    return total


def count_solutions_up_to(m: int) -> int:
    total = 0
    for c in range(1, m + 1):
        total += count_new_solutions(c)
    return total


def solve(target: int) -> int:
    """
    @formalsw: Py from project_euler_statements.p86 import naive
    @formalsw: Returns
    @formalsw: Equiv(naive)
    """
    total = 0
    c = 0
    while total <= target:
        c += 1
        total += count_new_solutions(c)
    return c


if __name__ == "__main__":
    # Given checks from the problem statement
    assert count_solutions_up_to(99) == 1975
    assert count_solutions_up_to(100) == 2060

    print(solve(1_000_000))
