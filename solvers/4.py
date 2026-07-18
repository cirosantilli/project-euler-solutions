#!/usr/bin/env python
from typing import Tuple


def solve(lo: int, hi: int) -> int:
    """
    Returns (best_palindrome, factor1, factor2) for factors in [lo, hi].
    """
    best = 0
    for a in range(hi, lo - 1, -1):
        # If even the maximum product with this 'a' can't beat best, we can stop.
        if a * hi < best:
            break
        for b in range(a, lo - 1, -1):
            prod = a * b
            # As b decreases, prod decreases; once <= best, no need to continue.
            if prod <= best:
                break
            s = str(prod)
            if s == s[::-1]:
                best = prod
                # For fixed a, this is the largest palindrome since b is descending.
                break
    return best


if __name__ == "__main__":
    assert solve(10, 99) == 9009
    print(solve(100, 999))
