#!/usr/bin/env python

def solve(n: int) -> int:
    s = n * (n + 1) // 2
    return (s * s) - (n * (n + 1) * (2 * n + 1) // 6)


if __name__ == "__main__":
    assert solve(10) == 2640
    print(solve(100))
