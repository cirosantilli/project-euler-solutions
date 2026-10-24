#!/usr/bin/env python

def is_triangle(triangle: list[list[int]]) -> bool:
    """
    Check that the triangle is nonempty and row lengths increase from one.
    """
    if not triangle:
        return False
    size = 1
    for row in triangle:
        if len(row) != size:
            return False
        size += 1
    return True


def solve(triangle: list[list[int]]) -> int:
    """
    @formalsw: Py from project_euler_statements.p18 import naive
    @formalsw: Requires(is_triangle(triangle))
    @formalsw: Equiv(naive)
    """
    dp = triangle[-1][:]  # best sums from current row to bottom
    for r in range(len(triangle) - 2, -1, -1):
        row = triangle[r]
        dp = [row[i] + max(dp[i], dp[i + 1]) for i in range(len(row))]
    return dp[0]


if __name__ == "__main__":
    # Sample from statement (answer = 23)
    sample = [
        [3],
        [7, 4],
        [2, 4, 6],
        [8, 5, 9, 3],
    ]
    assert solve(sample) == 23

    triangle = [
        [75],
        [95, 64],
        [17, 47, 82],
        [18, 35, 87, 10],
        [20, 4, 82, 47, 65],
        [19, 1, 23, 75, 3, 34],
        [88, 2, 77, 73, 7, 63, 67],
        [99, 65, 4, 28, 6, 16, 70, 92],
        [41, 41, 26, 56, 83, 40, 80, 70, 33],
        [41, 48, 72, 33, 47, 32, 37, 16, 94, 29],
        [53, 71, 44, 65, 25, 43, 91, 52, 97, 51, 14],
        [70, 11, 33, 28, 77, 73, 17, 78, 39, 68, 17, 57],
        [91, 71, 52, 38, 17, 14, 91, 43, 58, 50, 27, 29, 48],
        [63, 66, 4, 68, 89, 53, 67, 30, 73, 16, 69, 87, 40, 31],
        [4, 62, 98, 27, 23, 9, 70, 98, 73, 93, 38, 53, 60, 4, 23],
    ]

    print(solve(triangle))
