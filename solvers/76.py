#!/usr/bin/env python


def solve(n: int) -> int:
    """
    Count partitions of n using positive parts smaller than n.

    @formalsw: Py from project_euler_statements.p76 import naive
    @formalsw: Returns
    @formalsw: Equiv(naive)
    """
    if n < 0:
        return 0
    dp: list[int] = [0] * (n + 1)
    dp[0] = 1

    # Use parts 1..n-1, which excludes the single-part partition [n]
    for part in range(1, n):
        for s in range(part, n + 1):
            dp[s] += dp[s - part]

    return dp[n]


if __name__ == "__main__":
    # Given example: 5 can be written in 6 ways (excluding "5" itself)
    assert solve(5) == 6

    print(solve(100))
