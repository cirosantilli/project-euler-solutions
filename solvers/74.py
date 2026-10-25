#!/usr/bin/env python
from math import factorial


FACTORIALS = [factorial(digit) for digit in range(10)]


def next_term(n: int) -> int:
    if n == 0:
        return FACTORIALS[0]
    total = 0
    while n:
        n, digit = divmod(n, 10)
        total += FACTORIALS[digit]
    return total


def chain_length(start: int, lengths: dict[int, int]) -> int:
    """
    Memoize the number of distinct terms, sharing tails and complete cycles.
    """
    path = []
    positions = {}
    current = start
    while current not in lengths and current not in positions:
        positions[current] = len(path)
        path.append(current)
        current = next_term(current)

    if current in lengths:
        remaining = lengths[current]
        for value in reversed(path):
            remaining += 1
            lengths[value] = remaining
    else:
        cycle_start = positions[current]
        cycle_length = len(path) - cycle_start
        for value in path[cycle_start:]:
            lengths[value] = cycle_length
        remaining = cycle_length
        for value in reversed(path[:cycle_start]):
            remaining += 1
            lengths[value] = remaining
    return lengths[start]


def solve(limit: int = 1_000_000, target: int = 60) -> int:
    lengths = {}
    count = 0
    for start in range(1, limit):
        if chain_length(start, lengths) == target:
            count += 1
    return count


if __name__ == "__main__":
    lengths = {}
    assert chain_length(69, lengths) == 5
    assert chain_length(78, lengths) == 4
    assert chain_length(540, lengths) == 2
    assert chain_length(169, lengths) == 3
    print(solve())
