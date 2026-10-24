#!/usr/bin/env python
from typing import Tuple


def solve(digits: int) -> int:
    """
    @formalsw: Py from project_euler_statements.p25 import naive
    @formalsw: Requires(digits >= 1)
    @formalsw: Equiv(naive)

    Returns the smallest n such that Fibonacci number F_n has at least `digits` decimal digits.
    Fibonacci is defined as: F_1 = 1, F_2 = 1, F_n = F_{n-1} + F_{n-2}.
    """
    assert digits >= 1

    a, b = 1, 1  # F_1, F_2
    if digits == 1:
        return 1

    n = 2
    while len(str(b)) < digits:
        a, b = b, a + b
        n += 1
    return n


if __name__ == "__main__":
    # Tests implied by the statement / known small cases
    assert solve(1) == 1  # F_1 = 1
    assert solve(2) == 7  # F_7 = 13
    assert solve(3) == 12  # F_12 = 144

    print(solve(1000))
