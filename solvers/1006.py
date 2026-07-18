#!/usr/bin/env python
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache


MODULUS = 101_001_001
TARGET = 10**18
INV10 = pow(10, -1, MODULUS)
SMALL_LIMIT = 30


@dataclass(frozen=True)
class Macro:
    length: int
    total: int
    prefix: int
    square: int


@dataclass(frozen=True)
class Summary:
    length: int
    total: int
    prefix: int
    square: int


ZERO_MACRO = Macro(0, 0, 0, 0)
ZERO_SUMMARY = Summary(0, 0, 0, 0)

FIBS = [1, 2]
while FIBS[-1] <= TARGET:
    FIBS.append(FIBS[-1] + FIBS[-2])
FIB_INDEX = {value: index for index, value in enumerate(FIBS)}


def compose_macro(left: Macro, right: Macro) -> Macro:
    return Macro(
        (left.length + right.length) % MODULUS,
        (left.total + right.total) % MODULUS,
        (left.prefix + right.prefix + right.length * left.total) % MODULUS,
        (
            left.square
            + right.square
            + right.length * left.total * left.total
            + 2 * left.total * right.prefix
        )
        % MODULUS,
    )


def scale_macro(macro: Macro, factor: int) -> Macro:
    return Macro(
        macro.length,
        macro.total * factor % MODULUS,
        macro.prefix * factor % MODULUS,
        macro.square * factor * factor % MODULUS,
    )


def compose_summary(left: Summary, right: Summary) -> Summary:
    return Summary(
        (left.length + right.length) % MODULUS,
        (left.total + right.total) % MODULUS,
        (left.prefix + right.prefix + right.length * left.total) % MODULUS,
        (
            left.square
            + right.square
            + right.length * left.total * left.total
            + 2 * left.total * right.prefix
        )
        % MODULUS,
    )


def decomposition(size: int) -> tuple[str, str | int, int | None, str | None]:
    if size <= 1:
        return ("small", 0, None, None)

    fib_index = FIB_INDEX.get(size)
    if fib_index is not None:
        return ("boundary", "start" if fib_index % 2 else "end", size - 1, None)

    index = bisect_right(FIBS, size)
    modulus = FIBS[index]
    step = FIBS[index - 1] if index % 2 == 0 else FIBS[index - 2]
    complement = modulus - step
    base = max(step, complement)
    return ("insert", base, size - base, "after" if step > complement else "before")


@lru_cache(maxsize=None)
def small_permutation(size: int) -> tuple[int, ...]:
    if size == 0:
        return ()
    if size == 1:
        return (1,)

    kind, first, second, orientation = decomposition(size)
    if kind == "boundary":
        rest = small_permutation(second or 0)
        return (size,) + rest if first == "start" else rest + (size,)

    base = int(first)
    threshold = second or 0
    result: list[int] = []
    for label in small_permutation(base):
        if orientation == "before" and label <= threshold:
            result.append(label + base)
        result.append(label)
        if orientation == "after" and label <= threshold:
            result.append(label + base)
    return tuple(result)


def macro_at(pieces: tuple[tuple[int, int, Macro], ...], label: int) -> Macro:
    for low, high, macro in pieces:
        if low <= label <= high:
            return macro
    raise ValueError(f"missing macro for label {label}")


def merge_pieces(
    pieces: list[tuple[int, int, Macro]],
) -> tuple[tuple[int, int, Macro], ...]:
    merged: list[tuple[int, int, Macro]] = []
    for low, high, macro in pieces:
        if low > high:
            continue
        if merged and merged[-1][2] == macro and merged[-1][1] + 1 == low:
            merged[-1] = (merged[-1][0], high, macro)
        else:
            merged.append((low, high, macro))
    return tuple(merged)


def transform_pieces(
    size: int, pieces: tuple[tuple[int, int, Macro], ...]
) -> tuple[int, tuple[tuple[int, int, Macro], ...]]:
    kind, first, second, orientation = decomposition(size)
    assert kind == "insert"
    base = int(first)
    threshold = second or 0
    shifted_factor = pow(INV10, base, MODULUS)

    cuts = {1, base + 1, threshold + 1}
    for low, high, _ in pieces:
        left = max(1, low)
        right = min(base, high)
        if left <= right:
            cuts.add(left)
            cuts.add(right + 1)

        left = max(1, low - base)
        right = min(threshold, high - base)
        if left <= right:
            cuts.add(left)
            cuts.add(right + 1)

    result: list[tuple[int, int, Macro]] = []
    ordered_cuts = sorted(cut for cut in cuts if 1 <= cut <= base + 1)
    for low, next_cut in zip(ordered_cuts, ordered_cuts[1:]):
        high = next_cut - 1
        label = low
        base_macro = macro_at(pieces, label)
        shifted_macro = None
        if label <= threshold:
            shifted_macro = scale_macro(
                macro_at(pieces, label + base), shifted_factor
            )

        macro = ZERO_MACRO
        if orientation == "before" and shifted_macro is not None:
            macro = compose_macro(macro, shifted_macro)
        macro = compose_macro(macro, base_macro)
        if orientation == "after" and shifted_macro is not None:
            macro = compose_macro(macro, shifted_macro)
        result.append((low, high, macro))

    return base, merge_pieces(result)


def single_label_summary(
    label: int, pieces: tuple[tuple[int, int, Macro], ...], scale: int
) -> Summary:
    macro = macro_at(pieces, label)
    value = scale * pow(INV10, label, MODULUS) % MODULUS
    return Summary(
        macro.length,
        macro.total * value % MODULUS,
        macro.prefix * value % MODULUS,
        macro.square * value * value % MODULUS,
    )


def summarize(
    size: int, pieces: tuple[tuple[int, int, Macro], ...], scale: int
) -> Summary:
    if size == 0:
        return ZERO_SUMMARY
    if size <= SMALL_LIMIT:
        summary = ZERO_SUMMARY
        for label in small_permutation(size):
            summary = compose_summary(
                summary, single_label_summary(label, pieces, scale)
            )
        return summary

    kind, first, second, _ = decomposition(size)
    if kind == "boundary":
        boundary = single_label_summary(size, pieces, scale)
        rest = summarize(second or 0, pieces, scale)
        if first == "start":
            return compose_summary(boundary, rest)
        return compose_summary(rest, boundary)

    base, transformed = transform_pieces(size, pieces)
    return summarize(base, transformed, scale)


def prefix_before_label(
    size: int, pieces: tuple[tuple[int, int, Macro], ...], scale: int, target: int
) -> Summary:
    if size <= SMALL_LIMIT:
        summary = ZERO_SUMMARY
        for label in small_permutation(size):
            if label == target:
                return summary
            summary = compose_summary(
                summary, single_label_summary(label, pieces, scale)
            )
        raise ValueError(f"target label {target} not found")

    kind, first, second, orientation = decomposition(size)
    if kind == "boundary":
        rest_size = second or 0
        if target == size:
            if first == "start":
                return ZERO_SUMMARY
            return summarize(rest_size, pieces, scale)
        if first == "start":
            return compose_summary(
                single_label_summary(size, pieces, scale),
                prefix_before_label(rest_size, pieces, scale, target),
            )
        return prefix_before_label(rest_size, pieces, scale, target)

    base = int(first)
    threshold = second or 0
    _, transformed = transform_pieces(size, pieces)
    if target <= base:
        before = prefix_before_label(base, transformed, scale, target)
        if orientation == "before" and target <= threshold:
            before = compose_summary(
                before, single_label_summary(target + base, pieces, scale)
            )
        return before

    base_label = target - base
    before = prefix_before_label(base, transformed, scale, base_label)
    if orientation == "after":
        before = compose_summary(
            before, single_label_summary(base_label, pieces, scale)
        )
    return before


WORD_LENGTHS = [1, 2]
WORD_VALUES = [0, 1]
while WORD_LENGTHS[-1] <= TARGET:
    next_length = WORD_LENGTHS[-1] + WORD_LENGTHS[-2]
    next_value = (
        WORD_VALUES[-1] * pow(10, WORD_LENGTHS[-2], MODULUS) + WORD_VALUES[-2]
    ) % MODULUS
    WORD_LENGTHS.append(next_length)
    WORD_VALUES.append(next_value)


def fibonacci_prefix_value(length: int) -> int:
    result = 0
    remaining = length
    while remaining:
        index = bisect_right(WORD_LENGTHS, remaining) - 1
        result = (
            result * pow(10, WORD_LENGTHS[index], MODULUS) + WORD_VALUES[index]
        ) % MODULUS
        remaining -= WORD_LENGTHS[index]
    return result


def fibonacci_subword_square_sum(length: int) -> int:
    scale = 9 * pow(10, length - 1, MODULUS) % MODULUS
    initial_pieces = ((1, length, Macro(1, 1, 1, 1)),)
    generic = summarize(length, initial_pieces, scale)

    first_value = fibonacci_prefix_value(length - 1)
    answer = (length + 1) % MODULUS * first_value % MODULUS * first_value % MODULUS
    answer = (answer + 2 * first_value * generic.prefix + generic.square) % MODULUS

    generic_last_increment = 9 * INV10 % MODULUS
    correction = (1 - generic_last_increment) % MODULUS
    before_last = prefix_before_label(length, initial_pieces, scale, length)
    suffix_count = (length - before_last.length) % MODULUS
    suffix_prefix_sum = (generic.prefix - before_last.prefix) % MODULUS
    suffix_value_sum = (suffix_count * first_value + suffix_prefix_sum) % MODULUS
    return (
        answer
        + 2 * correction * suffix_value_sum
        + suffix_count * correction * correction
    ) % MODULUS


def main() -> None:
    assert fibonacci_subword_square_sum(10) == 10_699_667
    print(fibonacci_subword_square_sum(TARGET))


if __name__ == "__main__":
    main()
