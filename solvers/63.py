#!/usr/bin/env python


def solve(base: int = 10) -> int:
    """
    Count positive powers having as many base-b digits as their exponent.

    @formalsw: Requires(base >= 2)
    @formalsw: Py from project_euler_statements.p63 import naive
    @formalsw: Returns
    @formalsw: Equiv(naive)
    """
    total = 0
    for a in range(1, base):
        power = a
        lower = 1
        while power >= lower:
            total += 1
            power *= a
            lower *= base
    return total


if __name__ == "__main__":
    assert len(str(7 ** 5)) == 5
    assert len(str(8 ** 9)) == 9

    print(solve())
