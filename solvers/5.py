#!/usr/bin/env python

from math import gcd


def solve(n: int) -> int:
    ans = 1
    for x in range(2, n + 1):
        # lcm(ans, x)
        ans = ans // gcd(ans, x) * x
    return ans


if __name__ == "__main__":
    assert solve(10) == 2520
    print(solve(20))
