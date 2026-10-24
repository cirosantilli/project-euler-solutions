#!/usr/bin/env python
from __future__ import annotations

Signature = tuple[int, ...]


def digit_signature(x: int) -> Signature:
    cnt = [0] * 10
    for ch in str(x):
        cnt[ord(ch) - 48] += 1
    return tuple(cnt)


def solve(k: int) -> int:
    """
    Returns the smallest cube for which exactly k permutations of its digits are cubes.
    """
    groups: dict[Signature, list[int]] = {}  # sig -> [count, min_cube]
    current_len = 1

    n = 1
    while True:
        cube = n * n * n
        length = len(str(cube))

        if length > current_len:
            candidates = [
                min_cube for (count, min_cube) in groups.values() if count == k
            ]
            if candidates:
                return min(candidates)
            groups.clear()
            current_len = length

        sig = digit_signature(cube)
        if sig in groups:
            groups[sig][0] += 1
            if cube < groups[sig][1]:
                groups[sig][1] = cube
        else:
            groups[sig] = [1, cube]

        n += 1


if __name__ == "__main__":
    # Given example from the statement (Project Euler 62 known fact)
    assert solve(3) == 41063625

    print(solve(5))
