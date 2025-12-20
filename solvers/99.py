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


def best_line_number(pairs: List[Tuple[int, int]]) -> int:
    best_idx = -1
    best_score = float("-inf")
    for i, (base, exp) in enumerate(pairs, start=1):
        score = exp * math.log(base)  # compare via log(base^exp) = exp*log(base)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def find_input_file() -> str:
    return "0099_base_exp.txt"


def main() -> None:
    path = find_input_file()
    with open(path, "r", encoding="utf-8") as f:
        pairs = parse_pairs(f)
    assert best_line_number(pairs[:2]) == 2
    ans = best_line_number(pairs)
    print(ans)


if __name__ == "__main__":
    main()
