#!/usr/bin/env python
from __future__ import annotations

def is_rectangular(matrix: list[list[int]]) -> bool:
    if not matrix:
        return True
    width = len(matrix[0])
    for row in matrix:
        if len(row) != width:
            return False
    return True


def solve(matrix: list[list[int]]) -> int:
    """
    @formalsw: Requires(is_rectangular(matrix))
    @formalsw: Py from project_euler_statements.p81 import naive
    @formalsw: Returns
    @formalsw: Equiv(naive)
    """
    if not matrix:
        return 0
    assert is_rectangular(matrix)
    n = len(matrix)
    m = len(matrix[0])
    if m == 0:
        return 0

    row = matrix[0]
    dp = [0] * m
    dp[0] = row[0]

    # first row: only can come from left
    for j in range(1, m):
        dp[j] = dp[j - 1] + row[j]

    # remaining rows
    for i in range(1, n):
        row = matrix[i]
        # first col: only can come from top
        dp[0] = dp[0] + row[0]
        for j in range(1, m):
            dp[j] = row[j] + min(dp[j], dp[j - 1])
    return dp[-1]


def parse_matrix_from_text(text: str) -> list[list[int]]:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return [list(map(int, line.split(","))) for line in lines]


def read_matrix() -> list[list[int]]:
    with open("0081_matrix.txt", "r", encoding="utf-8") as f:
        return parse_matrix_from_text(f.read())


if __name__ == "__main__":
    sample = [
        [131, 673, 234, 103, 18],
        [201, 96, 342, 965, 150],
        [630, 803, 746, 422, 111],
        [537, 699, 497, 121, 956],
        [805, 732, 524, 37, 331],
    ]
    assert solve(sample) == 2427
    print(solve(read_matrix()))
