#!/usr/bin/env python
from typing import Set


def solve(max_a: int, max_b: int) -> int:
    """
    @formalsw: Py from project_euler_statements.p29 import naive
    @formalsw: Equiv(naive)
    """
    vals: Set[int] = set()
    for a in range(2, max_a + 1):
        for b in range(2, max_b + 1):
            vals.add(pow(a, b))
    return len(vals)


if __name__ == "__main__":
    # Test case from the problem statement: 2 <= a <= 5, 2 <= b <= 5 -> 15 distinct terms
    assert solve(5, 5) == 15

    print(solve(100, 100))
