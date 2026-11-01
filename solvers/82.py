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


def has_nonnegative_entries(matrix: list[list[int]]) -> bool:
    for row in matrix:
        for value in row:
            if value < 0:
                return False
    return True


def solve(matrix: list[list[int]]) -> int:
    """
    @formalsw: Requires(is_rectangular(matrix))
    @formalsw: Requires(has_nonnegative_entries(matrix))
    @formalsw: Py from project_euler_statements.p82 import naive
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

    # dp[i] = minimal cost to reach row i at current column
    dp = [matrix[i][0] for i in range(n)]

    for j in range(1, m):
        # Step 1: move right from previous column
        temp = [dp[i] + matrix[i][j] for i in range(n)]

        # Step 2: relax downward moves within this column
        for i in range(1, n):
            cand = temp[i - 1] + matrix[i][j]
            if cand < temp[i]:
                temp[i] = cand

        # Step 3: relax upward moves within this column
        for i in range(n - 2, -1, -1):
            cand = temp[i + 1] + matrix[i][j]
            if cand < temp[i]:
                temp[i] = cand

        dp = temp

    return min(dp)


def parse_matrix_from_text(text: str) -> list[list[int]]:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return [list(map(int, ln.split(","))) for ln in lines]


def read_input_matrix() -> list[list[int]]:
    with open("0082_matrix.txt", "r", encoding="utf-8") as f:
        return parse_matrix_from_text(f.read())


if __name__ == "__main__":
    sample = [
        [131, 673, 234, 103, 18],
        [201, 96, 342, 965, 150],
        [630, 803, 746, 422, 111],
        [537, 699, 497, 121, 956],
        [805, 732, 524, 37, 331],
    ]
    assert solve(sample) == 994
    print(solve(read_input_matrix()))
