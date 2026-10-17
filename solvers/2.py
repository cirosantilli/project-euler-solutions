#!/usr/bin/env python


def solve(limit: int) -> int:
    """
    @formalsw: Py from project_euler_statements.p2 import naive
    @formalsw: Requires(limit > 1)
    @formalsw: Equiv(naive)

    Returns the sum of even Fibonacci numbers not exceeding `limit`,
    with Fibonacci starting at 1, 2, ...
    """
    a, b = 1, 2
    total = 0
    while a <= limit:
        if a % 2 == 0:
            total += a
        a, b = b, a + b
    return total


if __name__ == "__main__":
    print(solve(4_000_000))
