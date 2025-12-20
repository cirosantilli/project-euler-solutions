#!/usr/bin/env python3
"""
Project Euler 719: Number Splitting

We define an S-number to be a natural number n that is a perfect square and whose
square root can be obtained by splitting the decimal representation of n into
2 or more numbers and adding them.

This script prints T(10^12) (or T(N) if N is given as a command-line argument).
"""

from __future__ import annotations

from math import isqrt
import sys


def digit_sum(n: int) -> int:
    s = 0
    while n:
        s += n % 10
        n //= 10
    return s


def is_s_number_root(root: int) -> bool:
    """
    Returns True if root^2 is an S-number.

    The check works by recursively splitting the square from the right:
      square = prefix * 10^k + suffix
    and trying to write root as suffix + (a valid split-sum of prefix).
    """
    sq = root * root
    if sq < 10:
        return False  # cannot be split into 2+ parts

    memo: dict[tuple[int, int], bool] = {}

    def dfs(num: int, target: int) -> bool:
        """
        Can the decimal digits of num be split into one or more parts that sum to target?
        """
        if target < 0 or target > num:
            return False
        if num == target:
            return True
        if num == 0:
            return target == 0

        # Necessary condition: any digit-splitting sum preserves value modulo 9.
        if (num - target) % 9 != 0:
            return False

        key = (num, target)
        cached = memo.get(key)
        if cached is not None:
            return cached

        # Lower bound: splitting into single digits gives the minimal possible sum.
        # This bound matters only when target is small, so avoid spending time
        # computing digit sums for large targets.
        if target < 120 and digit_sum(num) > target:
            memo[key] = False
            return False

        pow10 = 10
        while pow10 <= num:
            suffix = num % pow10
            if suffix > target:
                break  # longer suffixes only increase the numeric value
            prefix = num // pow10
            if dfs(prefix, target - suffix):
                memo[key] = True
                return True
            pow10 *= 10

        memo[key] = False
        return False

    # We must use at least one split overall. Enforce that by taking at least one
    # suffix at the top level (i.e. prefix must be non-zero here).
    pow10 = 10
    while pow10 <= sq:
        suffix = sq % pow10
        if suffix > root:
            break
        prefix = sq // pow10
        remaining = root - suffix
        if (prefix - remaining) % 9 == 0 and dfs(prefix, remaining):
            return True
        pow10 *= 10

    return False


def T(limit: int) -> int:
    """
    Sum of all S-numbers n <= limit.
    Since n must be a perfect square, we check roots up to isqrt(limit).
    """
    total = 0
    max_root = isqrt(limit)

    # From the modulo-9 invariant:
    #   root ≡ sum(parts of root^2) ≡ root^2 (mod 9)
    # which implies root mod 9 is 0 or 1.
    for base in (0, 1):
        start = 9 if base == 0 else 1
        for r in range(start, max_root + 1, 9):
            if r < 4:
                continue
            sq = r * r
            if sq <= limit and is_s_number_root(r):
                total += sq

    return total


def _self_test() -> None:
    # Examples from the problem statement:
    assert is_s_number_root(9)  # 81
    assert is_s_number_root(82)  # 6724
    assert is_s_number_root(91)  # 8281
    assert is_s_number_root(99)  # 9801

    # Given value:
    assert T(10**4) == 41333


def main(argv: list[str]) -> None:
    _self_test()

    limit = 10**12
    if len(argv) >= 2:
        limit = int(argv[1])

    print(T(limit))


if __name__ == "__main__":
    main(sys.argv)
