#!/usr/bin/env python
"""Project Euler 310: Nim Square."""

from __future__ import annotations

from math import isqrt


def build_grundy(limit: int) -> list[int]:
    squares = [i * i for i in range(1, isqrt(limit) + 1)]
    grundy = [0] * (limit + 1)
    seen = [0] * 128
    stamp = 0

    for size in range(1, limit + 1):
        stamp += 1
        for square in squares:
            if square > size:
                break
            g = grundy[size - square]
            if g >= len(seen):
                seen.extend([0] * len(seen))
            seen[g] = stamp

        mex = 0
        while seen[mex] == stamp:
            mex += 1
        grundy[size] = mex

    return grundy


def solve(limit: int) -> int:
    grundy = build_grundy(limit)

    size = 1
    while size <= max(grundy):
        size <<= 1

    prefix = [0] * size
    suffix = [0] * size
    for g in grundy:
        suffix[g] += 1

    total = 0
    for g in grundy:
        prefix[g] += 1
        for left, count in enumerate(prefix):
            if count:
                total += count * suffix[left ^ g]
        suffix[g] -= 1

    return total


def main() -> None:
    assert solve(4) == 12
    assert solve(29) == 1160
    print(solve(100000))


if __name__ == "__main__":
    main()
