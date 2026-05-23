#!/usr/bin/env python
from __future__ import annotations


def base7_digits(n: int) -> list[int]:
    if n == 0:
        return [0]

    digits = []
    while n:
        n, digit = divmod(n, 7)
        digits.append(digit)
    digits.reverse()
    return digits


def solve(num_rows: int) -> int:
    digits = base7_digits(num_rows)
    length = len(digits)

    powers28 = [1] * (length + 1)
    for i in range(1, length + 1):
        powers28[i] = powers28[i - 1] * 28

    total = 0
    prefix_product = 1
    for index, digit in enumerate(digits):
        remaining = length - index - 1
        smaller_digit_sum = digit * (digit + 1) // 2
        total += prefix_product * smaller_digit_sum * powers28[remaining]
        prefix_product *= digit + 1

    return total


def main() -> None:
    assert solve(7) == 28
    assert solve(100) == 2361
    print(solve(1_000_000_000))


if __name__ == "__main__":
    main()
