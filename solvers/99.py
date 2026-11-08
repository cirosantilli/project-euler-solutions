#!/usr/bin/env python
from __future__ import annotations

from math import gcd
from typing import Iterable, List, Tuple


def parse_pairs(lines: Iterable[str]) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        b_str, e_str = line.split(",")
        pairs.append((int(b_str), int(e_str)))
    return pairs


def round_power_bounds(lower: int, upper: int, shift: int, precision: int) -> tuple[int, int, int]:
    drop = max(0, upper.bit_length() - precision)
    return lower >> drop, (upper + (1 << drop) - 1) >> drop, shift + drop


def power_bounds(base: int, exponent: int, precision: int) -> tuple[int, int, int]:
    """Enclose base**exponent between lower*2**shift and upper*2**shift."""
    lower = 1
    upper = 1
    shift = 0
    factor_lower, factor_upper, factor_shift = round_power_bounds(base, base, 0, precision)
    while exponent:
        if exponent % 2:
            lower, upper, shift = round_power_bounds(
                lower * factor_lower, upper * factor_upper, shift + factor_shift, precision)
        exponent //= 2
        if exponent:
            factor_lower, factor_upper, factor_shift = round_power_bounds(
                factor_lower * factor_lower, factor_upper * factor_upper,
                2 * factor_shift, precision)
    return lower, upper, shift


def compare_scaled(left: int, left_shift: int, right: int, right_shift: int) -> int:
    if left == 0 or right == 0:
        return (left > right) - (left < right)
    left_size = left.bit_length() + left_shift
    right_size = right.bit_length() + right_shift
    if left_size != right_size:
        return (left_size > right_size) - (left_size < right_size)
    if left_shift >= right_shift:
        left <<= left_shift - right_shift
    else:
        right <<= right_shift - left_shift
    return (left > right) - (left < right)


def compare_powers(left_base: int, left_exp: int, right_base: int, right_exp: int) -> int:
    if left_base == 1:
        left_exp = 0
    if right_base == 1:
        right_exp = 0
    if left_exp == 0 and right_exp == 0:
        return 0
    if left_exp <= 0 and right_exp >= 0:
        return -1
    if left_exp >= 0 and right_exp <= 0:
        return 1
    if left_base == right_base:
        return (left_exp > right_exp) - (left_exp < right_exp)
    direction = 1
    if left_exp < 0:
        left_exp = -left_exp
        right_exp = -right_exp
        direction = -1
    common = gcd(left_exp, right_exp)
    left_exp //= common
    right_exp //= common
    precision = 16
    while True:
        lo, hi, shift = power_bounds(left_base, left_exp, precision)
        other_lo, other_hi, other_shift = power_bounds(right_base, right_exp, precision)
        if compare_scaled(lo, shift, other_hi, other_shift) > 0:
            return direction
        if compare_scaled(hi, shift, other_lo, other_shift) < 0:
            return -direction
        if lo == hi and other_lo == other_hi:
            return direction * compare_scaled(lo, shift, other_lo, other_shift)
        precision *= 2


def solve(pairs: List[Tuple[int, int]]) -> int:
    best_idx = 0
    best_base = 1
    best_exp = 0
    for index, (base, exponent) in enumerate(pairs, 1):
        if base < 1:
            raise ValueError('Bases must be positive')
        if best_idx == 0 or compare_powers(base, exponent, best_base, best_exp) > 0:
            best_idx = index
            best_base = base
            best_exp = exponent
    return best_idx


def find_input_file() -> str:
    return "0099_base_exp.txt"


if __name__ == "__main__":
    assert solve([]) == 0
    assert solve([(2, 3), (2, 3)]) == 1
    assert solve([(10**308, -10**308)]) == 1
    path = find_input_file()
    with open(path, "r", encoding="utf-8") as f:
        pairs = parse_pairs(f)
    assert solve(pairs[:2]) == 2
    print(solve(pairs))
