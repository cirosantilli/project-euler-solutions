#!/usr/bin/env python
from __future__ import annotations

import heapq
from pathlib import Path


def parse_matrix_from_text(text: str) -> list[list[int]]:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return [list(map(int, ln.split(","))) for ln in lines]


def load_matrix_file() -> list[list[int]]:
    return parse_matrix_from_text(Path("0083_matrix.txt").read_text(encoding="utf-8"))


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


def big_val(matrix: list[list[int]]) -> int:
    return sum(sum(row) for row in matrix) + 1


def solve(matrix: list[list[int]]) -> int:
    """
    @formalsw: Requires(is_rectangular(matrix))
    @formalsw: Requires(has_nonnegative_entries(matrix))
    @formalsw: Py from project_euler_statements.p83 import naive
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
    total = n * m

    infinity = big_val(matrix)
    dist = [infinity] * total
    start = 0
    goal = total - 1

    dist[start] = matrix[0][0]
    heap: list[tuple[int, int]] = [(matrix[0][0], start)]

    visited = [False] * total

    while heap:
        d, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        if u == goal:
            return d

        r, c = divmod(u, m)
        # 4-neighbors
        if r > 0:
            v = (r - 1) * m + c
            nd = d + matrix[r - 1][c]
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
        if r + 1 < n:
            v = (r + 1) * m + c
            nd = d + matrix[r + 1][c]
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
        if c > 0:
            v = r * m + c - 1
            nd = d + matrix[r][c - 1]
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
        if c + 1 < m:
            v = r * m + c + 1
            nd = d + matrix[r][c + 1]
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))

    raise RuntimeError("No path found (unexpected for a grid).")


if __name__ == "__main__":
    sample = """
    131,673,234,103,18
    201,96,342,965,150
    630,803,746,422,111
    537,699,497,121,956
    805,732,524,37,331
    """
    assert solve(parse_matrix_from_text(sample)) == 2297
    print(solve(load_matrix_file()))
