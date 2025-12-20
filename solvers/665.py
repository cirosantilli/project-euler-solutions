#!/usr/bin/env python3
"""
Project Euler 665: Proportionate Nim

We generate all losing positions (P-positions) for a two-heap impartial game with moves:
  - remove n from one heap
  - remove n from both heaps
  - remove n from one heap and 2n from the other

Let f(M) be the sum of n+m over losing positions (n,m) with n<=m and n+m<=M.
This program computes f(10^7).

No external libraries are used (only Python's standard library).
"""

from array import array
import sys


def _dsu_find(parent: array, x: int) -> int:
    """Return the smallest y >= x such that y is currently 'unused' (parent[y] == y)."""
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _dsu_mark_used(parent: array, x: int) -> None:
    """Mark x as used in the successor-DSU structure."""
    if parent[x] == x:
        parent[x] = _dsu_find(parent, x + 1)


def f(M: int) -> int:
    """
    Compute f(M) as defined in the problem statement.

    Strategy:
      - Build losing pairs (a,b) greedily by increasing the smallest unused coordinate a.
      - Choose the smallest b>a that avoids collisions in:
          * used coordinates
          * used differences (b-a)
          * used values of the linear form (x-2y), in both orientations:
                (b-2a) and (a-2b)
      - These collisions correspond exactly to the three move types.

    Implementation details:
      - Three successor-DSU structures are used to skip large blocks quickly:
          coords: next unused integer
          diff  : next unused difference
          v     : next unused value of x-2y (over a fixed bounded range)
      - The (a-2b) check is very rare; when it triggers we simply increment b by 1.
    """
    if M < 0:
        raise ValueError("M must be non-negative")

    # Only losing positions with n<=m and n+m<=M contribute, hence n <= M//2.
    half = M // 2

    # Empirically (and in practice for this problem), for n<=M/2 the partner m is < 1.125*M.
    # We keep a comfortable margin.
    max_coord = int(M * 1.25) + 100

    # Successor DSU for coordinates and differences (domain: 0..max_coord).
    coord_parent = array("I", range(max_coord + 2))
    diff_parent = array("I", range(max_coord + 2))

    # Successor DSU for values of v = x - 2y.
    # Over the range [-2*max_coord .. +max_coord].
    v_min = -2 * max_coord
    v_offset = -v_min  # maps v -> index via v+v_offset
    v_len = 3 * max_coord + 1  # number of valid v values
    v_parent = array("I", range(v_len + 1))  # extra sentinel slot at end

    def v_idx(v: int) -> int:
        return v + v_offset

    def v_find(i: int) -> int:
        while v_parent[i] != i:
            v_parent[i] = v_parent[v_parent[i]]
            i = v_parent[i]
        return i

    def v_mark(v: int) -> None:
        i = v_idx(v)
        if v_parent[i] == i:
            v_parent[i] = v_find(i + 1)

    # Base losing position (0,0); it doesn't affect sums but seeds the invariants.
    _dsu_mark_used(coord_parent, 0)
    _dsu_mark_used(diff_parent, 0)
    v_mark(0)

    total = 0
    a = 1

    while True:
        a = _dsu_find(coord_parent, a)
        if a > half:
            break

        b = a + 1
        while True:
            b = _dsu_find(coord_parent, b)

            d = b - a
            if diff_parent[d] != d:
                # Jump to the next unused difference >= d
                nd = _dsu_find(diff_parent, d)
                b = a + nd
                continue

            v1 = b - 2 * a
            i1 = v_idx(v1)
            if v_parent[i1] != i1:
                # Jump to the next unused v >= v1
                ni = v_find(i1)
                if ni >= v_len:
                    raise RuntimeError("Internal range for v was too small.")
                next_v = ni - v_offset
                b = 2 * a + next_v
                continue

            # Second orientation: a - 2b
            # This collision is rare; a +1 step is enough and keeps the code small.
            v2 = a - 2 * b
            i2 = v_idx(v2)
            if v_parent[i2] != i2:
                b += 1
                continue

            break

        # Record the new losing pair (a,b)
        _dsu_mark_used(coord_parent, a)
        _dsu_mark_used(coord_parent, b)
        _dsu_mark_used(diff_parent, d)
        v_mark(v1)
        v_mark(v2)

        s = a + b
        if s <= M:
            total += s

        a += 1

    return total


def _self_test() -> None:
    # Tests explicitly given in the problem statement:
    assert f(10) == 21
    assert f(100) == 1164
    assert f(1000) == 117002


def main(argv) -> None:
    _self_test()
    M = 10_000_000
    if len(argv) >= 2:
        M = int(argv[1])
    print(f(M))


if __name__ == "__main__":
    main(sys.argv)
