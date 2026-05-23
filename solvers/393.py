#!/usr/bin/env python
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache


FILLED = 0
HORIZONTAL = 1
VERTICAL = 2


def count_migrations(size: int) -> int:
    if size & 1:
        return 0

    full = (1 << size) - 1

    @lru_cache(maxsize=None)
    def transitions(state: int, last_row: bool) -> tuple[tuple[int, int], ...]:
        in1 = state & full
        in2 = state >> size
        counts: dict[int, int] = defaultdict(int)

        def options(occupied: int, out: int, col: int):
            bit = 1 << col
            if occupied & bit:
                yield FILLED, occupied, out
                return

            if col + 1 < size:
                next_bit = bit << 1
                if not (occupied & next_bit):
                    yield HORIZONTAL, occupied | bit | next_bit, out

            if not last_row:
                yield VERTICAL, occupied | bit, out | bit

        def fill_row(
            col: int,
            occupied1: int,
            occupied2: int,
            out1: int,
            out2: int,
        ) -> None:
            if col == size:
                if occupied1 == full and occupied2 == full:
                    counts[out1 | (out2 << size)] += 1
                return

            for kind1, next_occupied1, next_out1 in options(
                occupied1, out1, col
            ):
                for kind2, next_occupied2, next_out2 in options(
                    occupied2, out2, col
                ):
                    if kind1 != FILLED and kind1 == kind2:
                        continue
                    fill_row(
                        col + 1,
                        next_occupied1,
                        next_occupied2,
                        next_out1,
                        next_out2,
                    )

        fill_row(0, in1, in2, 0, 0)
        return tuple(counts.items())

    dp = {0: 1}
    for row in range(size):
        last_row = row + 1 == size
        next_dp: dict[int, int] = defaultdict(int)
        for state, ways in dp.items():
            for out_state, multiplicity in transitions(state, last_row):
                next_dp[out_state] += ways * multiplicity
        dp = dict(next_dp)
    return dp.get(0, 0)


def main() -> None:
    assert count_migrations(2) == 2
    assert count_migrations(3) == 0
    assert count_migrations(4) == 88
    print(count_migrations(10))


if __name__ == "__main__":
    main()
