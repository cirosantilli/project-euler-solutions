#!/usr/bin/env python
from math import isqrt


def period_length_sqrt(n: int) -> int:
    """
    Return the continued-fraction period length of sqrt(n), or zero for squares.
    """
    a0 = isqrt(n)
    if a0 * a0 == n:
        return 0

    m = 0
    d = 1
    a = a0

    period = 0
    while True:
        m = d * a - m
        d = (n - m * m) // d
        a = (a0 + m) // d
        period += 1
        if a == 2 * a0:
            return period


def solve(limit: int) -> int:
    count = 0
    for n in range(2, limit + 1):
        r = isqrt(n)
        if r * r == n:
            continue
        if period_length_sqrt(n) % 2 == 1:
            count += 1
    return count


if __name__ == "__main__":
    assert solve(13) == 4
    print(solve(10_000))
