#!/usr/bin/env python

def solve(n: int) -> int:
    """
    @formalsw: Py from project_euler_statements.p6 import naive
    @formalsw: Requires(n >= 0)
    @formalsw: Equiv(naive)
    """
    s = n * (n + 1) // 2
    return (
        # Square of sum
        (s * s) -
        # Sum of squares
        (n * (n + 1) * (2 * n + 1) // 6)
    )


if __name__ == "__main__":
    assert solve(10) == 2640
    print(solve(100))
