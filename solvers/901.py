#!/usr/bin/env python

from __future__ import annotations

import math


MAX_DEPTH = 80.0
MAX_STEPS = 200


def is_admissible(first_depth: float) -> bool:
    previous = 0.0
    current = first_depth

    for _ in range(MAX_STEPS):
        gap = current - previous
        if gap <= 0.0:
            return False
        if current > MAX_DEPTH or gap > MAX_DEPTH:
            return True
        previous, current = current, math.exp(gap)

    return True


def find_upper_branch_boundary() -> float:
    previous_x = 0.6
    previous_ok = is_admissible(previous_x)

    x = 0.61
    while x <= 1.5 + 1e-12:
        ok = is_admissible(x)
        if not previous_ok and ok:
            low, high = previous_x, x
            break
        previous_x, previous_ok = x, ok
        x += 0.01
    else:
        raise RuntimeError("failed to bracket the upper admissible branch")

    for _ in range(90):
        midpoint = (low + high) / 2
        if is_admissible(midpoint):
            high = midpoint
        else:
            low = midpoint
    return high


def expected_cost(first_depth: float) -> float:
    previous = 0.0
    current = first_depth
    total = first_depth + 1.0

    for _ in range(MAX_STEPS):
        total += math.exp(-current)
        if current > MAX_DEPTH:
            break
        previous, current = current, math.exp(current - previous)

    return total


def solve() -> float:
    return expected_cost(find_upper_branch_boundary())


def main() -> None:
    print(f"{solve():.9f}")


if __name__ == "__main__":
    main()
