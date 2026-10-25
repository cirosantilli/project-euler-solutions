#!/usr/bin/env python
from pathlib import Path


def parse_triangle(text: str) -> list[list[int]]:
    rows: list[list[int]] = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([int(x) for x in line.split()])
    return rows


def is_triangle(triangle: list[list[int]]) -> bool:
    size = 1
    for row in triangle:
        if len(row) != size:
            return False
        size += 1
    return True


def solve(triangle: list[list[int]]) -> int:
    """
    Compute the greatest path sum, returning zero for an empty triangle.

    @formalsw: Py from project_euler_statements.p67 import naive
    @formalsw: Requires(is_triangle(triangle))
    @formalsw: Returns
    @formalsw: Equiv(naive)
    """
    if not triangle:
        return 0
    # Bottom-up DP: dp[j] holds best sum from current position to bottom.
    dp = triangle[-1][:]  # copy last row
    for i in range(len(triangle) - 2, -1, -1):
        row = triangle[i]
        for j in range(len(row)):
            dp[j] = row[j] + max(dp[j], dp[j + 1])
    return dp[0]


def read_triangle_file() -> str:
    return Path("0067_triangle.txt").read_text(encoding="utf-8")


if __name__ == "__main__":
    # Sample from the statement
    sample = """
    3
    7 4
    2 4 6
    8 5 9 3
    """
    assert solve(parse_triangle(sample)) == 23

    print(solve(parse_triangle(read_triangle_file())))
