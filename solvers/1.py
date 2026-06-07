#!/usr/bin/env python


def sum_of_multiples(m: int, n: int) -> int:
    max_range = n // m
    return m * max_range * (max_range + 1) // 2


def solve(n: int) -> int:
    n -= 1
    return sum_of_multiples(3, n) + sum_of_multiples(5, n) - sum_of_multiples(15, n)


if __name__ == "__main__":
    assert solve(10) == 23
    print(solve(1000))
