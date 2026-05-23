#!/usr/bin/env python
from __future__ import annotations

from functools import lru_cache

MOD = 998244353
TOP = -1
ODD = -2


def boundary_is_impossible(state: int) -> bool:
    return state == 0


def satisfies(state: int, value: int) -> bool:
    if state == TOP:
        return True
    if state == ODD:
        return (value & 1) == 1
    return (value & state) != 0


def phi_even(state: int) -> int:
    if state == TOP:
        return TOP
    if state == ODD:
        return 0
    return state // 2


def phi_odd(state: int) -> int:
    if state == TOP or state == ODD:
        return TOP
    if state & 1:
        return TOP
    return state // 2


MAX_N = 123
FIB = [0] * (MAX_N + 2)
FIB[0] = 0
FIB[1] = 1
for i in range(2, len(FIB)):
    FIB[i] = (FIB[i - 1] + FIB[i - 2]) % MOD


def fib(index: int) -> int:
    if index >= 0:
        return FIB[index]
    index = -index
    return FIB[index] if index & 1 else (-FIB[index]) % MOD


@lru_cache(maxsize=None)
def D(length: int, bound: int, left: int, right: int) -> int:
    if boundary_is_impossible(left) or boundary_is_impossible(right):
        return 0

    if length == 0:
        return 1 if left == TOP and right == TOP else 0

    if bound <= 1:
        if length == 1:
            return sum(
                1
                for value in range(bound + 1)
                if satisfies(left, value) and satisfies(right, value)
            )
        if bound == 0:
            return 0
        return 1 if satisfies(left, 1) and satisfies(right, 1) else 0

    if bound % 2 == 0:
        marked_bound = bound
        total = D(length, bound - 1, left, right)

        # Add sequences whose first occurrence of the exceptional top value
        # `bound` is at position split.
        for split in range(1, length + 1):
            prefix_len = split - 1
            suffix_len = length - split

            if prefix_len == 0:
                prefix_count = 1 if satisfies(left, bound) else 0
            else:
                prefix_count = D(prefix_len, bound - 1, left, marked_bound)
            if prefix_count == 0:
                continue

            if suffix_len == 0:
                suffix_count = 1 if satisfies(right, bound) else 0
            else:
                suffix_count = D(suffix_len, bound, marked_bound, right)

            total = (total + prefix_count * suffix_count) % MOD

        return total

    reduced_bound = (bound - 1) // 2
    left_even = phi_even(left)
    left_odd = phi_odd(left)
    right_even = phi_even(right)
    right_odd = phi_odd(right)

    total = D(length, reduced_bound, left_even, right_even) * fib(length)
    total += D(length, reduced_bound, left_even, right_odd) * fib(length - 1)
    total += D(length, reduced_bound, left_odd, right_even) * fib(length - 1)
    total += D(length, reduced_bound, left_odd, right_odd) * fib(length - 2)
    total %= MOD

    # First adjacent odd-odd pair. That edge is already satisfied by the low
    # bit, so the suffix can be counted independently at the original bound.
    for cut in range(1, length):
        prefix = D(cut, reduced_bound, left_odd, TOP) * fib(cut - 2)
        if cut > 1:
            prefix += D(cut, reduced_bound, left_even, TOP) * fib(cut - 1)
        prefix %= MOD
        if prefix:
            total = (total + D(length - cut, bound, ODD, right) * prefix) % MOD

    return total


def c(length: int, bound: int) -> int:
    return D(length, bound, TOP, TOP)


def main() -> None:
    assert c(3, 4) == 18
    assert c(10, 6) == 2496120
    assert c(100, 200) == 268159379
    print(c(123, 123456789))


if __name__ == "__main__":
    main()
