#!/usr/bin/env python
from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict


def coefficients(limit: int) -> tuple[list[int], list[int], list[int]]:
    """Return boundary, tail, and value coefficients for singleton bits."""
    h = [0] * (limit + 4)
    h[1] = 0
    h[2] = 0
    h[3] = 1
    for index in range(4, limit + 1):
        h[index] = 2 * h[index - 3] - h[index - 2]

    boundary = [0] * limit
    tail = [0] * limit
    value = [0] * limit
    for pos in range(3, limit):
        boundary[pos] = h[pos] - 3 * h[pos - 1] + 2 * h[pos - 2]
        tail[pos] = h[pos - 1] - 2 * h[pos - 2]
        value[pos] = 2 * (tail[pos] + h[pos])
    return boundary, tail, value


def allowed_prefix_masks(last_mask: int) -> list[int]:
    """Return right-half first-two-bit masks compatible with the left edge."""
    allowed = []
    for prefix in range(4):
        if (last_mask & 1) and (prefix & 3):
            continue
        if (last_mask & 2) and (prefix & 1):
            continue
        allowed.append(prefix)
    return allowed


def sad_sum(limit: int) -> int:
    """Return S(limit)."""
    boundary, tail, value = coefficients(limit)
    split = limit // 2

    right: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)

    def generate_right(
        pos: int,
        prev1: int,
        prev2: int,
        boundary_sum: int,
        tail_sum: int,
        value_sum: int,
        prefix_mask: int,
    ) -> None:
        if pos == limit:
            right[(boundary_sum, prefix_mask)].append((tail_sum, value_sum))
            return

        generate_right(
            pos + 1,
            0,
            prev1,
            boundary_sum,
            tail_sum,
            value_sum,
            prefix_mask,
        )
        if prev1 or prev2:
            return

        next_prefix = prefix_mask
        if pos == split:
            next_prefix |= 1
        elif pos == split + 1:
            next_prefix |= 2
        generate_right(
            pos + 1,
            1,
            prev1,
            boundary_sum + boundary[pos],
            tail_sum + tail[pos],
            value_sum + value[pos],
            next_prefix,
        )

    generate_right(split, 0, 0, 0, 0, 0, 0)

    indexed_right: dict[tuple[int, int], tuple[list[int], list[int]]] = {}
    for key, entries in right.items():
        entries.sort()
        tails = [tail_sum for tail_sum, _ in entries]
        prefix_values = [0]
        running = 0
        for _, value_sum in entries:
            running += value_sum
            prefix_values.append(running)
        indexed_right[key] = (tails, prefix_values)

    prefix_options = [allowed_prefix_masks(mask) for mask in range(4)]
    total = 0

    def generate_left(
        pos: int,
        prev1: int,
        prev2: int,
        boundary_sum: int,
        tail_sum: int,
        value_sum: int,
    ) -> None:
        nonlocal total
        if pos == split:
            last_mask = prev1 | (prev2 << 1)
            needed_boundary = -boundary_sum
            needed_tail = -tail_sum
            for prefix_mask in prefix_options[last_mask]:
                packed = indexed_right.get((needed_boundary, prefix_mask))
                if packed is None:
                    continue
                tails, prefix_values = packed
                index = bisect_left(tails, needed_tail)
                count = len(tails) - index
                if count:
                    total += count * value_sum + prefix_values[-1] - prefix_values[index]
            return

        generate_left(pos + 1, 0, prev1, boundary_sum, tail_sum, value_sum)
        if prev1 or prev2:
            return

        add_boundary = 0
        add_tail = 0
        add_value = 0
        if pos == 0:
            add_value = 1
        elif pos == 1:
            add_boundary = -1
        elif pos == 2:
            add_boundary = 1
            add_tail = -1
            add_value = -2
        else:
            add_boundary = boundary[pos]
            add_tail = tail[pos]
            add_value = value[pos]
        generate_left(
            pos + 1,
            1,
            prev1,
            boundary_sum + add_boundary,
            tail_sum + add_tail,
            value_sum + add_value,
        )

    generate_left(0, 0, 0, 0, 0, 0)
    return total


def main() -> None:
    assert sad_sum(14) == 159
    assert sad_sum(30) == 33438
    print(sad_sum(80))


if __name__ == "__main__":
    main()
