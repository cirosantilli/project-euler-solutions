#!/usr/bin/env python
from __future__ import annotations


def polygonal(s: int, n: int) -> int:
    """
    @formalsw: Returns
    @formalsw: Py from project_euler_statements.p61 import polygonal as reference_polygonal
    @formalsw: Equiv(reference_polygonal)
    s-gonal number P_{s,n} = n((s-2)n - (s-4)) / 2, for s >= 3
    """
    return n * ((s - 2) * n - (s - 4)) // 2


def generate_4digit_polygonals(s: int) -> list[int]:
    nums: list[int] = []
    n = 1
    while True:
        val = polygonal(s, n)
        if val >= 10000:
            break
        if val >= 1000:
            nums.append(val)
        n += 1
    return nums


def build_prefix_map(nums: list[int]) -> dict[int, list[int]]:
    mp: dict[int, list[int]] = {}
    for x in nums:
        pref = x // 100
        mp.setdefault(pref, []).append(x)
    return mp


def solve(max_sides: int = 8) -> int:
    """
    Find the largest cycle sum using one distinct four-digit number of each type 3..max_sides.

    @formalsw: Returns
    @formalsw: Py from project_euler_statements.p61 import naive
    @formalsw: Equiv(naive)
    """
    if max_sides < 3:
        return 0
    types = list(range(3, max_sides + 1))
    nums_by_type = {t: generate_4digit_polygonals(t) for t in types}
    pref_map = {t: build_prefix_map(nums_by_type[t]) for t in types}

    def dfs(path: list[int], remaining: list[int]) -> int:
        if not remaining:
            if path[-1] % 100 == path[0] // 100:
                return sum(path)
            return 0
        best = 0
        suffix = path[-1] % 100
        for t in remaining:
            unused = [other for other in remaining if other != t]
            for nxt in pref_map[t].get(suffix, []):
                if nxt in path:
                    continue
                best = max(best, dfs(path + [nxt], unused))
        return best

    best = 0
    for start in nums_by_type[3]:
        best = max(best, dfs([start], list(range(4, max_sides + 1))))
    return best


if __name__ == "__main__":
    # Problem statement examples
    assert polygonal(3, 127) == 8128
    assert polygonal(4, 91) == 8281
    assert polygonal(5, 44) == 2882

    print(solve())
