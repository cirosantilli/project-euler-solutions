#!/usr/bin/env python
from __future__ import annotations

MODULUS = 1_000_000_007
ALPHABET_SIZE = 10
MAX_SIDE = 10
MAX_DIGITS = MAX_SIDE * MAX_SIDE


def partitions(max_part: int, max_len: int):
    current: list[int] = []

    def rec(limit: int, remaining_len: int):
        if current:
            yield tuple(current)
        if remaining_len == 0:
            return
        for part in range(limit, 0, -1):
            current.append(part)
            yield from rec(part, remaining_len - 1)
            current.pop()

    yield from rec(max_part, max_len)


def shape_weight(shape: tuple[int, ...], factorials: list[int]) -> int:
    """Return f^shape * s_shape(1^10) modulo MODULUS."""
    column_heights = [
        sum(1 for row_length in shape if row_length >= col)
        for col in range(1, shape[0] + 1)
    ]
    result = factorials[sum(shape)]
    for row_index, row_length in enumerate(shape, start=1):
        for col_index in range(1, row_length + 1):
            hook = (
                row_length
                - col_index
                + column_heights[col_index - 1]
                - row_index
                + 1
            )
            content_factor = ALPHABET_SIZE + col_index - row_index
            result = result * content_factor % MODULUS
            result = result * pow(hook, MODULUS - 3, MODULUS) % MODULUS
    return result


def count_balanced_positive(max_digits: int = MAX_DIGITS) -> int:
    factorials = [1] * (MAX_DIGITS + 1)
    for value in range(1, MAX_DIGITS + 1):
        factorials[value] = factorials[value - 1] * value % MODULUS

    balanced_all = 0
    balanced_ending_nine = 1 if max_digits >= 1 else 0
    for shape in partitions(MAX_SIDE, MAX_SIDE):
        size = sum(shape)
        width = shape[0]
        height = len(shape)
        weight = shape_weight(shape, factorials)

        if size <= max_digits and width == height:
            balanced_all = (balanced_all + weight) % MODULUS
        if size + 1 <= max_digits and height == width + 1:
            balanced_ending_nine = (balanced_ending_nine + weight) % MODULUS

    return (balanced_all - balanced_ending_nine) % MODULUS


def main() -> None:
    assert count_balanced_positive(4) == 2274
    print(count_balanced_positive())


if __name__ == "__main__":
    main()
