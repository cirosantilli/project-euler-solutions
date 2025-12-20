#!/usr/bin/env python3
# Project Euler 917
#
# We have an N x N matrix M where M[i,j] = a_i + b_j.
# The path sum is over cells from (1,1) to (N,N) with moves Right/Down.
#
# Key reduction:
#   Each a_i appears exactly once on the path (because every row is visited),
#   plus once for each right-move made while in row i.
#   Similarly each b_j appears exactly once, plus once for each down-move in column j.
#
# Therefore:
#   A(N) = (sum_i a_i) + (sum_j b_j) + D(N)
# where D(N) is the shortest path cost in a grid graph on nodes (i,j), 1..N,
# with directed edges:
#   (i,j) -> (i, j+1) cost a_i
#   (i,j) -> (i+1, j) cost b_j
#
# The full DP is O(N^2). The special structure lets us prune rows/columns:
# Only indices that are vertices of the *lower convex hull* of points (i, a_i)
# and (j, b_j) can be turning points of an optimal path. After keeping only those
# rows/columns, the remaining graph is tiny; we run a standard DP on the compressed grid.
#
# No external libraries are used.

from __future__ import annotations

MOD = 998388889
S1 = 102022661


def compute_A(N: int) -> int:
    """Compute A(N) for the problem definition."""
    if N <= 0:
        raise ValueError("N must be positive")

    mod = MOD
    s = S1

    sum_a = 0
    sum_b = 0

    # Lower convex hulls for points (i, a_i) and (i, b_i), stored as parallel stacks.
    ax: list[int] = []
    ay: list[int] = []
    bx: list[int] = []
    by: list[int] = []

    # Local bindings for speed in the 10^7 loop.
    ax_append = ax.append
    ay_append = ay.append
    bx_append = bx.append
    by_append = by.append
    ax_pop = ax.pop
    ay_pop = ay.pop
    bx_pop = bx.pop
    by_pop = by.pop

    for i in range(1, N + 1):
        a = s
        s = (s * s) % mod
        b = s
        s = (s * s) % mod

        sum_a += a
        sum_b += b

        # Update lower hull for (i, a_i).
        while len(ax) >= 2:
            x1 = ax[-2]
            y1 = ay[-2]
            x2 = ax[-1]
            y2 = ay[-1]
            # cross((x2-x1, y2-y1), (i-x2, a-y2)) <= 0  => pop
            if (x2 - x1) * (a - y2) - (y2 - y1) * (i - x2) <= 0:
                ax_pop()
                ay_pop()
            else:
                break
        ax_append(i)
        ay_append(a)

        # Update lower hull for (i, b_i).
        while len(bx) >= 2:
            x1 = bx[-2]
            y1 = by[-2]
            x2 = bx[-1]
            y2 = by[-1]
            if (x2 - x1) * (b - y2) - (y2 - y1) * (i - x2) <= 0:
                bx_pop()
                by_pop()
            else:
                break
        bx_append(i)
        by_append(b)

    # DP on the compressed grid formed by hull vertices.
    rows_x, rows_cost = ax, ay
    cols_x, cols_cost = bx, by

    R = len(rows_x)
    C = len(cols_x)

    INF = 10**60
    dp = [INF] * C
    dp[0] = 0

    # First row: only moves right.
    a0 = rows_cost[0]
    prev_cx = cols_x[0]
    for j in range(1, C):
        cx = cols_x[j]
        dp[j] = dp[j - 1] + a0 * (cx - prev_cx)
        prev_cx = cx

    # Remaining rows.
    for i in range(1, R):
        new = [INF] * C
        dr = rows_x[i] - rows_x[i - 1]

        # First column: only moves down.
        new[0] = dp[0] + cols_cost[0] * dr

        ai = rows_cost[i]
        for j in range(1, C):
            # Down from (i-1, j)
            down = dp[j] + cols_cost[j] * dr
            # Right from (i, j-1)
            dc = cols_x[j] - cols_x[j - 1]
            right = new[j - 1] + ai * dc
            new[j] = down if down < right else right

        dp = new

    D = dp[-1]
    return sum_a + sum_b + D


def _self_test() -> None:
    # Test values from the problem statement.
    assert compute_A(1) == 966774091
    assert compute_A(2) == 2388327490
    assert compute_A(10) == 13389278727


def main() -> None:
    _self_test()
    print(compute_A(10_000_000))


if __name__ == "__main__":
    main()
