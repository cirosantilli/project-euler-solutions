#!/usr/bin/env python3
"""
Project Euler 736: Paths to Equality

We work with:
    r(x,y) = (x+1, 2y)
    s(x,y) = (2x, y+1)

Goal: for (45,90), find the unique path to equality with smallest odd length,
and output the final value v where the path ends at (v,v).

No external libraries are used (only Python standard library).
"""

from __future__ import annotations

from itertools import combinations_with_replacement
from typing import Iterable, List, Sequence, Tuple


Point = Tuple[int, int]


def apply_ops(start: Point, ops: Sequence[str]) -> List[Point]:
    """Apply ops (each 'r' or 's') starting from `start`, returning all visited points (including start)."""
    x, y = start
    states: List[Point] = [(x, y)]
    for op in ops:
        if op == "r":
            x, y = x + 1, y * 2
        elif op == "s":
            x, y = x * 2, y + 1
        else:
            raise ValueError(f"Unknown op: {op!r}")
        states.append((x, y))
    return states


def is_path_to_equality(states: Sequence[Point]) -> bool:
    """True iff final point has x==y and all earlier points have x!=y."""
    if not states:
        return False
    for x, y in states[:-1]:
        if x == y:
            return False
    x, y = states[-1]
    return x == y


def brute_exists_shorter_path(start: Point, max_steps: int) -> bool:
    """
    Brute-force: check whether there exists ANY path-to-equality from `start`
    using <= max_steps steps. (Tiny only; used for the sample assertion.)
    """
    a, b = start
    for steps in range(max_steps + 1):
        # iterate over all 2**steps sequences via bitmasks
        for mask in range(1 << steps):
            ops = []
            for i in range(steps):
                ops.append("r" if (mask >> i) & 1 else "s")
            states = apply_ops((a, b), ops)
            if is_path_to_equality(states):
                return True
    return False


def rhs_from_positions(t: int, positions: Sequence[int]) -> int:
    """
    For fixed t, and a sorted multiset `positions` (values in [0, t-1]) of size s,
    compute:
        RHS = sum_{j=0..t-1} 2^{P_j}
    where P_j = count(positions <= j).
    """
    s = len(positions)
    idx = 0
    total = 0
    for j in range(t):
        while idx < s and positions[idx] <= j:
            idx += 1
        total += 1 << idx
    return total


def find_min_odd_path_value() -> Tuple[int, int, Sequence[str]]:
    """
    Returns:
        (steps, final_value, ops)
    where ops is the unique smallest-odd-length path's forward operation list.
    """
    start = (45, 90)

    # For even step count k, a size argument shows that for k < 96, we must have
    # exactly the same number of s and r steps (p=q=t), hence k=2t.
    #
    # For (45,90) with b=2a, the equality condition for p=q=t collapses to a small
    # combinatorial identity once we force the 2^t-leading coefficient, leaving
    # only s = t-45 "early" r-steps to locate. We search t upward.
    #
    # The search is tiny: t=45..48 and s=t-45 <= 3 for the first solution.
    for t in range(45, 200):
        s = t - 45
        if s < 0:
            continue

        sols: List[Tuple[int, ...]] = []
        for pos in combinations_with_replacement(range(t), s):
            lhs = sum(1 << p for p in pos)  # sum of 2^{column} for the s early r-steps
            rhs = rhs_from_positions(
                t, pos
            )  # sum of 2^{prefix count} for the t s-steps
            if lhs == rhs:
                sols.append(pos)

        if not sols:
            continue

        # The problem statement guarantees uniqueness for the smallest odd length; verify it.
        assert (
            len(sols) == 1
        ), f"Expected a unique solution at minimal t, got {len(sols)}"
        pos = sols[0]

        # Build counts c_j: number of reverse-R steps taken in column j (0..t).
        # For the minimal solution, the last column has exactly 45 reverse-R steps;
        # the s=t-45 remaining reverse-R steps occur in columns listed by `pos`.
        c = [0] * (t + 1)
        for p in pos:
            c[p] += 1
        c[t] = 45

        # Build reverse operation list from (v,v) back to (45,90):
        # For each column j=0..t-1: do R c[j] times, then one S; finally do R c[t] times.
        # Here:
        #   R: (x,y)->(x-1, y/2)   (inverse of forward r)
        #   S: (x,y)->(x/2, y-1)   (inverse of forward s)
        ops_rev: List[str] = []
        for j in range(t):
            ops_rev.extend(["R"] * c[j])
            ops_rev.append("S")
        ops_rev.extend(["R"] * c[t])

        assert len(ops_rev) == 2 * t

        # Convert to forward ops (reverse the list and invert each step).
        inv = {"R": "r", "S": "s"}
        ops_fwd = [inv[o] for o in reversed(ops_rev)]

        states = apply_ops(start, ops_fwd)
        assert len(states) == 2 * t + 1
        assert is_path_to_equality(
            states
        ), "Constructed path is not a valid path to equality"

        final_value = states[-1][0]
        return (2 * t, final_value, ops_fwd)

    raise RuntimeError("No solution found in the searched range.")


def _run_sample_asserts() -> None:
    # Sample path from the problem statement:
    # (45,90) ->r ->s ->s ->s ->s ->r ->s ->r ->r -> (1476,1476)
    sample_ops = list("rssssrsrr")
    sample_states = apply_ops((45, 90), sample_ops)
    assert len(sample_states) == 10
    assert is_path_to_equality(sample_states)
    assert sample_states[-1] == (1476, 1476)

    # And there is no shorter path to equality (i.e., <=8 steps).
    assert not brute_exists_shorter_path((45, 90), max_steps=8)


def main() -> None:
    _run_sample_asserts()
    steps, final_value, _ops = find_min_odd_path_value()
    # Requirement: do not assert the final answer; just print it.
    print(final_value)


if __name__ == "__main__":
    main()
