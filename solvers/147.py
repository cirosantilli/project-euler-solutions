#!/usr/bin/env python
"""Adapted from: https://github.com/stbrumme/euler/blob/b426763514558c3b39f2ec507f271d322088d28a/euler-0147.cpp"""


def grid(width: int, height: int) -> int:
    return (width * (width + 1) // 2) * (height * (height + 1) // 2)


def capped_descending_sum(cap: int, start: int, last_index: int) -> int:
    first_descending = min(last_index, start - cap)
    total = 0

    if first_descending >= 0:
        total += (first_descending + 1) * cap

    if last_index > first_descending:
        count = last_index - first_descending
        lo = first_descending + 1
        hi = last_index
        total += count * start - (lo + hi) * count // 2

    return total


def diagonal(width: int, height: int, cache) -> int:
    key = (width, height)
    if key in cache:
        return cache[key]

    a, b = width, height
    if a < b:
        a, b = b, a

    count = 0
    for i in range(a):
        for j in range(b):
            for parity in (0, 1):
                start_x = 2 * i + 1 + parity
                start_y = 2 * j + 2 - parity
                x_limit = 2 * a - start_x
                y_limit = 2 * b - start_y
                cap = min(x_limit, y_limit)
                last_width = min(start_y - 1, x_limit - 1)

                if cap > 0 and last_width >= 0:
                    count += capped_descending_sum(cap, x_limit, last_width)

    cache[key] = count
    return count


def count_grid(width: int, height: int, cache) -> int:
    return grid(width, height) + diagonal(width, height, cache)


def solve(max_width: int, max_height: int) -> int:
    sum_upright = 0
    sum_diagonal = 0
    cache: dict[tuple[int, int], int] = {}
    for width in range(1, max_width + 1):
        for height in range(1, max_height + 1):
            sum_upright += grid(width, height)
            sum_diagonal += diagonal(width, height, cache)
    return sum_upright + sum_diagonal


def main() -> None:
    cache: dict[tuple[int, int], int] = {}
    assert count_grid(3, 2, cache) == 37
    assert solve(3, 2) == 72
    print(solve(47, 43))


if __name__ == "__main__":
    main()
