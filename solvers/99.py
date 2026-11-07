#!/usr/bin/env python
from __future__ import annotations

import math
from typing import Iterable, List, Tuple


def parse_pairs(lines: Iterable[str]) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        b_str, e_str = line.split(",")
        pairs.append((int(b_str), int(e_str)))
    return pairs


def solve(pairs: List[Tuple[int, int]]) -> int:
    best_idx = 0
    best_score = float("-inf")
    for i, (base, exp) in enumerate(pairs, start=1):
        score = exp * math.log(base)  # compare via log(base^exp) = exp*log(base)
        if best_idx == 0 or score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def find_input_file() -> str:
    return "0099_base_exp.txt"


if __name__ == "__main__":
    assert solve([]) == 0
    assert solve([(2, 3), (2, 3)]) == 1
    assert solve([(10**308, -10**308)]) == 1
    path = find_input_file()
    with open(path, "r", encoding="utf-8") as f:
        pairs = parse_pairs(f)
    assert solve(pairs[:2]) == 2
    print(solve(pairs))
