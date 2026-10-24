#!/usr/bin/env python
from __future__ import annotations

from math import isqrt
from typing import List


def pentagonal(n: int) -> int:
    return n * (3 * n - 1) // 2


def is_pentagonal(x: int) -> bool:
    # x is pentagonal iff 24x+1 is a perfect square s^2 and (1+s) divisible by 6
    if x <= 0:
        return False
    t = 24 * x + 1
    s = isqrt(t)
    return s * s == t and (1 + s) % 6 == 0


def solve(limit: int = 10_000) -> int:
    pent: List[int] = [0] * (limit + 1)
    for n in range(1, limit + 1):
        pent[n] = pentagonal(n)

    best = 10**30
    for k in range(2, limit + 1):
        pk = pent[k]
        # j desc: diff increases as j decreases, so we can break once diff >= best
        for j in range(k - 1, 0, -1):
            diff = pk - pent[j]
            if diff >= best:
                break
            if is_pentagonal(diff) and is_pentagonal(pk + pent[j]):
                best = diff
    return best


if __name__ == "__main__":
    first10 = [1, 5, 12, 22, 35, 51, 70, 92, 117, 145]
    for i, v in enumerate(first10, start=1):
        assert pentagonal(i) == v
        assert is_pentagonal(v)

    assert pentagonal(4) + pentagonal(7) == pentagonal(8)  # 22 + 70 = 92
    assert not is_pentagonal(pentagonal(7) - pentagonal(4))  # 48 is not pentagonal
    print(solve(10_000))
