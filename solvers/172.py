#!/usr/bin/env python


def solve(length: int, max_use: int) -> int:
    if length <= 0 or max_use <= 0:
        return 0

    factorials = [1] * (length + 1)
    for i in range(1, length + 1):
        factorials[i] = factorials[i - 1] * i

    base = factorials[length - 1]
    total = 0

    def search(digit: int, remaining: int, denominator: int, zero_count: int) -> None:
        nonlocal total
        if digit == 10:
            if remaining == 0:
                total += (length - zero_count) * base // denominator
            return

        remaining_digits = 10 - digit
        if remaining > remaining_digits * max_use:
            return

        limit = min(max_use, remaining)
        for count in range(limit + 1):
            search(
                digit + 1,
                remaining - count,
                denominator * factorials[count],
                count if digit == 0 else zero_count,
            )

    search(0, length, 1, 0)
    return total


if __name__ == "__main__":
    print(solve(18, 3))
