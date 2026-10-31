#!/usr/bin/env python
from __future__ import annotations

from itertools import combinations


SQUARE_PAIRS: list[tuple[int, int]] = [
    (0, 1),  # 01
    (0, 4),  # 04
    (0, 9),  # 09
    (1, 6),  # 16
    (2, 5),  # 25
    (3, 6),  # 36
    (4, 9),  # 49
    (6, 4),  # 64
    (8, 1),  # 81
]


def expand_6_9(mask: int) -> int:
    """
    Treat either printed 6 or 9 as displaying both rotated digits.
    """
    has6 = (mask >> 6) & 1
    has9 = (mask >> 9) & 1
    if has6 or has9:
        mask |= (1 << 6) | (1 << 9)
    return mask


def can_display(mask_a: int, mask_b: int, required_pairs: list[tuple[int, int]]) -> bool:
    for x, y in required_pairs:
        bx = 1 << x
        by = 1 << y
        if not ((mask_a & bx and mask_b & by) or (mask_a & by and mask_b & bx)):
            return False
    return True


def solve(face_count: int, required_pairs: list[tuple[int, int]]) -> int:
    if face_count < 0 or face_count > 10:
        return 0
    masks: list[int] = []
    for comb in combinations(range(10), face_count):
        m = 0
        for d in comb:
            m |= 1 << d
        masks.append(m)

    expanded: list[int] = [expand_6_9(m) for m in masks]

    count = 0
    n = len(masks)
    for i in range(n):
        for j in range(i, n):  # unordered pairs, allow i == j
            if can_display(expanded[i], expanded[j], required_pairs):
                count += 1
    return count


if __name__ == "__main__":
    print(solve(6, SQUARE_PAIRS))
