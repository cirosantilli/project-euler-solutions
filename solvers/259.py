#!/usr/bin/env python
"""
Project Euler 259: Reachable Numbers

Use the digits 1..9 in order, allowing concatenation, binary arithmetic
operations, and arbitrary parentheses.  Sum the distinct positive integers that
can be reached exactly.
"""

import multiprocessing as mp
import os
from math import gcd


DIGITS = "123456789"
ADD = 0
SUB = 1
MUL = 2
DIV = 3
CONCAT = 4


def normalize(num: int, den: int) -> tuple[int, int]:
    if num == 0:
        return 0, 1
    if den < 0:
        num = -num
        den = -den
    g = gcd(abs(num), den)
    return num // g, den // g


def add_combined(
    out: set[tuple[int, int]],
    op: int,
    left_values: set[tuple[int, int]],
    right_values: set[tuple[int, int]],
) -> None:
    add = out.add
    common_den: dict[tuple[int, int], tuple[int, int]] = {}

    if op == ADD:
        for an, ad in left_values:
            for bn, bd in right_values:
                key = (ad, bd)
                muls = common_den.get(key)
                if muls is None:
                    g = gcd(ad, bd)
                    muls = (bd // g, ad // g, ad // g * bd)
                    common_den[key] = muls
                left_mul, right_mul, den = muls
                add(normalize(an * left_mul + bn * right_mul, den))
    elif op == SUB:
        for an, ad in left_values:
            for bn, bd in right_values:
                key = (ad, bd)
                muls = common_den.get(key)
                if muls is None:
                    g = gcd(ad, bd)
                    muls = (bd // g, ad // g, ad // g * bd)
                    common_den[key] = muls
                left_mul, right_mul, den = muls
                add(normalize(an * left_mul - bn * right_mul, den))
    elif op == MUL:
        for an, ad in left_values:
            for bn, bd in right_values:
                if an == 0 or bn == 0:
                    add((0, 1))
                    continue
                g1 = gcd(abs(an), bd)
                g2 = gcd(abs(bn), ad)
                add(((an // g1) * (bn // g2), (ad // g2) * (bd // g1)))
    else:
        for an, ad in left_values:
            for bn, bd in right_values:
                if bn:
                    g1 = gcd(abs(an), abs(bn))
                    g2 = gcd(bd, ad)
                    num = (an // g1) * (bd // g2)
                    den = (ad // g2) * (bn // g1)
                    add(normalize(num, den))


def decode_pattern(digits: str, code: int) -> tuple[list[int], list[int]]:
    values: list[int] = []
    ops: list[int] = []
    current = ord(digits[0]) - ord("0")

    for digit in digits[1:]:
        choice = code % 5
        code //= 5
        next_value = ord(digit) - ord("0")
        if choice == CONCAT:
            current = 10 * current + next_value
        else:
            values.append(current)
            ops.append(choice)
            current = next_value

    values.append(current)
    return values, ops


def all_results(values: list[int], ops: list[int]) -> set[tuple[int, int]]:
    count = len(values)
    dp: list[list[set[tuple[int, int]]]] = [
        [set() for _ in range(count)] for _ in range(count)
    ]

    for i, value in enumerate(values):
        dp[i][i].add((value, 1))

    for width in range(2, count + 1):
        for lo in range(0, count - width + 1):
            hi = lo + width - 1
            results: set[tuple[int, int]] = set()
            for split in range(lo, hi):
                add_combined(results, ops[split], dp[lo][split], dp[split + 1][hi])
            dp[lo][hi] = results

    return dp[0][count - 1]


def dedupe_sorted(values: list[int]) -> list[int]:
    if not values:
        return []
    values.sort()
    out = [values[0]]
    last = values[0]
    for value in values[1:]:
        if value != last:
            out.append(value)
            last = value
    return out


def collect_for_range(args: tuple[str, int, int]) -> list[int]:
    digits, start, stop = args
    found: list[int] = []
    append = found.append

    for code in range(start, stop):
        values, ops = decode_pattern(digits, code)
        for num, den in all_results(values, ops):
            if den == 1 and num > 0:
                append(num)

    return dedupe_sorted(found)


def default_workers(total_patterns: int) -> int:
    env = os.environ.get("PE259_PROCS", "").strip()
    if env:
        try:
            return max(1, min(total_patterns, int(env)))
        except ValueError:
            pass
    return max(1, min(8, os.cpu_count() or 1, total_patterns))


def reachable_integers(digits: str = DIGITS, workers: int | None = None) -> set[int]:
    total_patterns = 5 ** (len(digits) - 1)
    if workers is None:
        workers = default_workers(total_patterns) if digits == DIGITS else 1

    if workers <= 1 or total_patterns == 1:
        return set(collect_for_range((digits, 0, total_patterns)))

    chunk = (total_patterns + workers - 1) // workers
    tasks = []
    for start in range(0, total_patterns, chunk):
        tasks.append((digits, start, min(total_patterns, start + chunk)))

    integers: set[int] = set()
    try:
        with mp.Pool(processes=len(tasks)) as pool:
            for values in pool.imap_unordered(collect_for_range, tasks, chunksize=1):
                integers.update(values)
    except Exception:
        integers = set(collect_for_range((digits, 0, total_patterns)))

    return integers


def solve(last_digit: int = 9) -> int:
    return sum(reachable_integers(DIGITS[:last_digit]))


def main() -> None:
    integers = reachable_integers(DIGITS)
    assert 42 in integers

    print(sum(integers))


if __name__ == "__main__":
    main()
