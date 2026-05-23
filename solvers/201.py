#!/usr/bin/env python
"""Project Euler 201: Subsets with a Unique Sum."""

from __future__ import annotations


def unique_sum_from_list(values: list[int], choose: int) -> int:
    if choose < 0 or choose > len(values):
        return 0
    if choose == 0:
        return 0

    max_sum = sum(sorted(values)[-choose:])
    missing = max_sum + 1

    # state[j][s] is 0, 1, or 2, where 2 means "at least two ways".
    state = [bytearray(max_sum + 1) for _ in range(choose + 1)]
    low = [missing] * (choose + 1)
    high = [-1] * (choose + 1)
    state[0][0] = 1
    low[0] = high[0] = 0

    used = 0
    for square in values:
        upper = min(choose, used + 1)
        for size in range(upper, 0, -1):
            lo = low[size - 1]
            if lo == missing:
                continue

            hi = high[size - 1]
            src = state[size - 1]
            dst = state[size]
            for subtotal in range(hi, lo - 1, -1):
                ways = src[subtotal]
                if ways == 0:
                    continue
                new_sum = subtotal + square
                new_ways = dst[new_sum] + ways
                dst[new_sum] = 2 if new_ways >= 2 else new_ways

            new_lo = lo + square
            new_hi = hi + square
            if new_lo < low[size]:
                low[size] = new_lo
            if new_hi > high[size]:
                high[size] = new_hi
        used += 1

    final = state[choose]
    return sum(total for total in range(low[choose], high[choose] + 1) if final[total] == 1)


def solve(max_set: int, choose: int) -> int:
    return unique_sum_from_list([i * i for i in range(1, max_set + 1)], choose)


def main() -> None:
    assert unique_sum_from_list([1, 3, 6, 8, 10, 11], 3) == 156
    assert solve(5, 3) == 330
    print(solve(100, 50))


if __name__ == "__main__":
    main()
