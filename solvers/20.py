#!/usr/bin/env python


def solve(n: int) -> int:
    """
    @formalsw: Py from project_euler_statements.p20 import naive
    @formalsw: Requires(n >= 0)
    @formalsw: Equiv(naive)
    """
    fact = 1
    for k in range(2, n + 1):
        fact *= k
    return sum(int(ch) for ch in str(fact))


if __name__ == "__main__":
    # Test case from the problem statement
    assert solve(10) == 27

    print(solve(100))
