#!/usr/bin/env python


def digit_sum(n: int) -> int:
    return sum(int(ch) for ch in str(n))


def solve(n: int = 1000) -> int:
    """
    @formalsw: Requires(n >= 0)
    @formalsw: Py from project_euler_statements.p16 import naive
    @formalsw: Equiv(naive)
    """
    return digit_sum(2**n)


def _tests() -> None:
    # Example from the statement
    assert digit_sum(2**15) == 26


if __name__ == "__main__":
    _tests()
    print(solve())
