#!/usr/bin/env python
from typing import List


def solve(target: int, coins: List[int]) -> int:
    """
    @formalsw: Py from project_euler_statements.p31 import count_ways_bound
    @formalsw: Requires(target >= 0 and all(c > 0 for c in coins))
    @formalsw: Equiv(count_ways_bound)
    """
    dp = [0] * (target + 1)
    dp[0] = 1
    for c in coins:
        for s in range(c, target + 1):
            dp[s] += dp[s - c]
    return dp[target]


if __name__ == "__main__":
    coins = [1, 2, 5, 10, 20, 50, 100, 200]

    # Basic sanity checks (no official test cases were given in the statement)
    assert solve(0, coins) == 1
    assert solve(1, coins) == 1
    assert solve(2, coins) == 2
    assert solve(5, coins) == 4
    assert solve(10, coins) == 11


    print(solve(200, coins))
