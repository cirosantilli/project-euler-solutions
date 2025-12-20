#!/usr/bin/env python3
"""
Project Euler 693: Finite Sequence Generator

We define a sequence from integers x > y > 0:
    a_x = y
    a_{z+1} = a_z^2 mod z  for z = x, x+1, ...
and stop when a term becomes 0 or 1.
Let l(x,y) be the number of terms produced.

For each x, g(x) = max_{1 <= y < x} l(x,y).
For each n, f(n) = max_{2 <= x <= n} g(x).
This program computes f(3_000_000).

No external libraries are used.
"""

from __future__ import annotations

import heapq
import math
from typing import Dict, List


def l_xy(x: int, y: int) -> int:
    """Directly simulate l(x,y) for a single starting value y (used for small checks)."""
    z = x
    a = y
    length = 1
    while a not in (0, 1):
        a = (a * a) % z
        z += 1
        length += 1
    return length


def _initial_active_after_first_step(x: int) -> List[int]:
    """
    After the first transition (mod x), take the union over all y in {2,3,...,x-1}
    of y^2 mod x, then discard terminal values 0 and 1.

    We exploit symmetry: y^2 mod x == (x-y)^2 mod x, so iterating y=2..x//2
    already covers all non-terminal images.
    """
    if x <= 2:
        return []

    mark = bytearray(x)
    active: List[int] = []

    # incremental squares modulo x for y = 2..x//2
    y = 2
    sq = (y * y) % x
    delta = 2 * y + 1  # (y+1)^2 - y^2
    limit = x // 2

    while y <= limit:
        if sq > 1 and not mark[sq]:
            mark[sq] = 1
            active.append(sq)

        # advance to next square modulo x without using % in the common case
        sq += delta
        if sq >= x:
            sq -= x
            if sq >= x:  # extremely rare, but safe
                sq -= x
        delta += 2
        y += 1

    return active


def _step_active_mark(active: List[int], mod: int) -> List[int]:
    """One transition using a bytearray marker for fast deduplication."""
    mark = bytearray(mod)
    nxt: List[int] = []
    for a in active:
        v = (a * a) % mod
        if v > 1 and not mark[v]:
            mark[v] = 1
            nxt.append(v)
    return nxt


def _step_active_set(active: List[int], mod: int) -> List[int]:
    """One transition using a Python set (best once the active set is small)."""
    s = set()
    for a in active:
        v = (a * a) % mod
        if v > 1:
            s.add(v)
    return list(s)


def g(x: int, big_threshold: int = 100_000) -> int:
    """
    Compute g(x) (maximum chain length over y<x).

    Key idea:
      Track the *active* set of values (excluding 0 and 1) that can appear at each index z
      for some starting y. Because we take a union over all y, this active set evolves as:
          A_{z+1} = { a^2 mod z : a in A_z } \ {0,1}
      The process ends exactly when every possible starting value has reached 0 or 1,
      i.e. when A becomes empty. The number of generated terms equals the number of steps
      taken plus 1 for the initial term.

    Implementation notes:
      - We start after the first step (mod x), because the initial values {2..x-1}
        are huge; the first image can be computed directly.
      - While the active set is large we deduplicate with a bytearray marker.
      - As soon as it collapses to a singleton, we follow that single value forward
        without building sets (critical for long chains).
    """
    if x < 2:
        return 0
    if x == 2:
        # Only y=1 is allowed and it's terminal immediately.
        return 1

    active = _initial_active_after_first_step(x)
    length = 2  # we've produced terms at indices x and x+1
    mod = x + 1

    # Reduce until empty or singleton
    while len(active) > 1:
        if len(active) >= big_threshold:
            active = _step_active_mark(active, mod)
        else:
            active = _step_active_set(active, mod)

        length += 1
        mod += 1

        if not active:
            return length

    if not active:
        return length

    # Singleton fast path: just follow the chain
    v = active[0]
    while v not in (0, 1):
        v = (v * v) % mod
        length += 1
        mod += 1

    return length


def f(n: int, target_points: int = 16) -> int:
    """
    Compute f(n) = max_{2 <= x <= n} g(x).

    We use a best-first branch-and-bound search.

    Upper bound:
      For any r >= x, dropping the first (r-x) terms of any sequence starting at x
      yields a valid sequence starting at r, so:
          g(x) <= g(r) + (r - x)
      Therefore, for an interval [l, r]:
          max_{l <= x <= r} g(x) <= g(r) + (r - l)

    Strategy:
      1) Evaluate g at a coarse grid of points to get a decent lower bound quickly.
      2) Put each gap [l, r] into a max-heap keyed by the upper bound.
      3) Pop the most promising interval; if its bound can't beat the best, stop.
      4) Otherwise split at the midpoint, evaluate g(mid), and push sub-intervals.

    This is exact: every pruned interval provably cannot contain a better value.
    """
    if n < 2:
        return 0

    cache: Dict[int, int] = {}

    def G(x: int) -> int:
        v = cache.get(x)
        if v is None:
            v = g(x)
            cache[x] = v
        return v

    # Choose a coarse grid: about `target_points` points.
    if target_points < 2:
        target_points = 2
    grid = max(1, (n - 2) // (target_points - 1))

    # Snap grid to a "nice" 1/2/5 * 10^k value to reduce jitter across runs.
    if grid > 1:
        p = 10 ** int(math.log10(grid))
        for m in (1, 2, 5, 10):
            if m * p >= grid:
                grid = m * p
                break

    points = list(range(2, n + 1, grid))
    if points[-1] != n:
        points.append(n)

    best = 0
    g_at_right: Dict[int, int] = {}
    for x in points:
        gx = G(x)
        g_at_right[x] = gx
        if gx > best:
            best = gx

    # Max-heap via negative keys: (-(upper_bound), l, r, g(r))
    heap = []
    for i in range(1, len(points)):
        l = points[i - 1]
        r = points[i]
        gr = g_at_right[r]
        ub = gr + (r - l)
        heapq.heappush(heap, (-ub, l, r, gr))

    while heap:
        ub = -heap[0][0]
        if ub <= best:
            break

        _, l, r, gr = heapq.heappop(heap)
        if r - l <= 1:
            continue

        m = (l + r) // 2
        gm = G(m)
        if gm > best:
            best = gm

        # Left interval [l, m] uses g(m) as its right-end value
        if m - l > 1:
            ub_left = gm + (m - l)
            if ub_left > best:
                heapq.heappush(heap, (-ub_left, l, m, gm))

        # Right interval [m, r] keeps g(r)
        if r - m > 1:
            ub_right = gr + (r - m)
            if ub_right > best:
                heapq.heappush(heap, (-ub_right, m, r, gr))

    return best


def _self_test() -> None:
    # Test values explicitly stated in the problem statement.
    assert l_xy(5, 3) == 29
    assert g(5) == 29
    assert f(100) == 145
    assert f(10_000) == 8824


def main() -> None:
    _self_test()
    print(f(3_000_000))


if __name__ == "__main__":
    main()
