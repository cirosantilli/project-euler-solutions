#!/usr/bin/env python

from math import factorial


TARGET = tuple(range(1, 25)) + (
    26,
    25,
    27,
    28,
    30,
    32,
    34,
    36,
    38,
    40,
    39,
    42,
    45,
    43,
    41,
    37,
    35,
    33,
    31,
    29,
    44,
)


def first_sort_count(permutation: tuple[int, ...]) -> int:
    positions = [0] * (len(permutation) + 1)
    for index, value in enumerate(permutation):
        positions[value] = index

    total = 0
    maxima_mask = 1
    for value in range(2, len(permutation) + 1):
        insertion_pos = 1
        value_pos = positions[value]
        for smaller in range(1, value):
            if positions[smaller] < value_pos:
                insertion_pos += 1

        low_mask = 1 << (insertion_pos - 1)
        total += maxima_mask - (maxima_mask % low_mask)
        maxima_mask = low_mask + (maxima_mask % low_mask)

    return total


def lex_index(permutation: tuple[int, ...]) -> int:
    available = list(range(1, len(permutation) + 1))
    rank = 1
    for index, value in enumerate(permutation):
        smaller_count = available.index(value)
        rank += smaller_count * factorial(len(permutation) - index - 1)
        available.pop(smaller_count)
    return rank


def main() -> None:
    assert first_sort_count((4, 1, 3, 2)) == 5
    assert first_sort_count(TARGET) == 12**12
    print(lex_index(TARGET))


if __name__ == "__main__":
    main()
