#!/usr/bin/env python
from __future__ import annotations


def digit_sum(n: int) -> int:
    return sum(ord(ch) - 48 for ch in str(n))


def max_digital_sum(limit: int) -> tuple[int, int, int]:
    """
    Return (best_sum, best_a, best_b) for 0 <= a,b < limit.

    The empty search returns (0, -1, -1).
    """
    best = 0
    best_a = -1
    best_b = -1
    for a in range(limit):
        for b in range(limit):
            s = digit_sum(pow(a, b))
            if s > best:
                best = s
                best_a = a
                best_b = b
    return best, best_a, best_b


def solve(limit: int) -> int:
    return max_digital_sum(limit)[0]


if __name__ == "__main__":
    # Basic sanity checks
    assert digit_sum(10**100) == 1
    assert digit_sum(2**15) == 26
    print(solve(100))
