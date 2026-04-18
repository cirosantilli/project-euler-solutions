#!/usr/bin/env python
from __future__ import annotations

# Eventual periodicity of the first-difference sequence BB(n+1) - BB(n):
# starting at n = 514, the deltas repeat with period 71.
PERIOD_START = 514
PERIOD = 71
DELTA_PATTERN = [
    17, -2, -8, -2, -2, -14, -2, -2, -17, -8, -5, -8, -5, -2, -2, -5, -8,
    50, -8, 23, -13, -2, 67, -5, -2, -2, -5, -8, -5, 21, 29, -11, -2, -2,
    6, -11, 31, -2, -11, 17, -2, -8, -2, -2, -14, -2, -2, -17, -8, -5, -8,
    -8, 8, -13, -5, -2, -2, -5, -2, -11, -8, -8, -5, -2, -11, -8, -8, -5,
    -2, -11, 216,
]
PATTERN_SUM = sum(DELTA_PATTERN)


def step_state(pos: int, carry: int, bananas: set[int]) -> tuple[int, int, set[int]] | None:
    """Apply one game step.

    Returns the updated state, or None if the game halts before making a move.
    """
    has_x = pos in bananas
    has_x1 = (pos + 1) in bananas

    if has_x and has_x1:
        bananas = set(bananas)
        bananas.remove(pos + 1)
        return pos - 1, carry + 1, bananas

    if has_x and not has_x1:
        bananas = set(bananas)
        bananas.remove(pos)
        return pos + 2, carry + 1, bananas

    if (not has_x) and has_x1:
        bananas = set(bananas)
        bananas.remove(pos + 1)
        bananas.add(pos)
        return pos + 2, carry, bananas

    if carry >= 3:
        bananas = set(bananas)
        bananas.add(pos - 1)
        bananas.add(pos)
        bananas.add(pos + 1)
        return pos - 2, carry - 3, bananas

    return None


def simulate_steps(initial_bananas: int, steps: int) -> tuple[int, int, set[int]]:
    """Simulate a fixed number of steps from the empty-line start."""
    pos = 0
    carry = initial_bananas
    bananas: set[int] = set()
    for _ in range(steps):
        nxt = step_state(pos, carry, bananas)
        if nxt is None:
            break
        pos, carry, bananas = nxt
    return pos, carry, bananas


def simulate_bb_values(limit: int) -> list[int]:
    """Directly simulate BB(0), BB(1), ..., BB(limit)."""
    bb = [0]
    pos = 0
    carry = 0
    bananas: set[int] = set()

    for _n in range(1, limit + 1):
        carry += 1
        while True:
            nxt = step_state(pos, carry, bananas)
            if nxt is None:
                bb.append(pos)
                break
            pos, carry, bananas = nxt
    return bb


def build_prefix() -> list[int]:
    # We only need values up to PERIOD_START plus one full period for verification.
    return simulate_bb_values(PERIOD_START + PERIOD)


def bb(n: int, bb_prefix: list[int]) -> int:
    if n <= PERIOD_START:
        return bb_prefix[n]

    remaining = n - PERIOD_START
    whole_periods, tail = divmod(remaining, PERIOD)
    return bb_prefix[PERIOD_START] + whole_periods * PATTERN_SUM + sum(DELTA_PATTERN[:tail])


if __name__ == "__main__":
    bb_prefix = build_prefix()

    # Problem-statement checks.
    pos, carry, bananas = simulate_steps(3, 1)
    assert pos == -2
    assert carry == 0
    assert bananas == {-1, 0, 1}

    pos, carry, bananas = simulate_steps(5, 5)
    assert pos == -1
    assert carry == 0
    assert bananas == {-2, -1, 0, 1, 2}

    assert bb(1000, bb_prefix) == 1499

    # Verify that the recorded eventual period matches direct simulation.
    deltas = [bb_prefix[i + 1] - bb_prefix[i] for i in range(len(bb_prefix) - 1)]
    for i, delta in enumerate(DELTA_PATTERN):
        assert deltas[PERIOD_START + i] == delta

    print(bb(10**18, bb_prefix))
