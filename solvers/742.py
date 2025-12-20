#!/usr/bin/env python3
"""
Project Euler 742: Minimum Area of a Convex Grid Polygon

We search for the minimum area A(N) among symmetrical convex grid polygons with N vertices.

Key constraints:
- integer-coordinate vertices
- strictly convex
- horizontal + vertical symmetry

The solver uses a standard reduction: the polygon is determined by a set of primitive direction
vectors (a,b) in the first quadrant, duplicated by symmetry.  The main difficulty is choosing
which primitive vectors minimize the area for a fixed vertex count.

A practical and fast approach is to consider that optimal sets correspond to primitive vectors
inside an axis-aligned ellipse.  This reduces the problem to selecting k primitive vectors with
minimum weight a^2 + t*b^2 for some ellipse aspect parameter t, then scanning t over a fine grid.

This implementation:
- scans t from 0.001 .. 1.000 (inclusive) at step 0.001 (symmetry lets us restrict to t<=1)
- selects the best candidate set of k vectors for each t
- constructs the symmetric polygon (in edge-vector form)
- computes the area using an O(m) prefix-determinant method (m ~ N/2)

Asserts are included for test values given in the problem statement.
"""

import math
from typing import List, Tuple


Pair = Tuple[int, int]


def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def primitive_pairs(limit: int) -> List[Tuple[int, int, int, int]]:
    """Return primitive (a,b) with 1<=a,b<=limit as tuples (a,b,a^2,b^2)."""
    out = []
    for a in range(1, limit + 1):
        a2 = a * a
        for b in range(1, limit + 1):
            if gcd(a, b) == 1:
                out.append((a, b, a2, b * b))
    return out


def n_smallest_by_weight(pairs, k: int, t: float) -> List[Pair]:
    """
    Select k pairs (a,b) minimizing a^2 + t*b^2.
    Uses a size-k max-heap (implemented via negative weights).
    """
    # heap stores (-weight, -tie1, -a, -b, a, b)
    # tie1 = a+b ensures stable preference toward smaller L1 if weights match.
    heap = []

    def push_item(a, b, a2, b2):
        w = a2 + t * b2
        tie1 = a + b
        item = (-w, -tie1, -a, -b, a, b)
        if len(heap) < k:
            heap.append(item)
        else:
            # replace worst if better
            if item > heap[0]:
                heap[0] = item

    # build as list then heapify once for speed
    for a, b, a2, b2 in pairs:
        if len(heap) < k:
            w = a2 + t * b2
            tie1 = a + b
            heap.append((-w, -tie1, -a, -b, a, b))
        else:
            break

    import heapq

    heapq.heapify(heap)

    for a, b, a2, b2 in pairs[len(heap) :]:
        w = a2 + t * b2
        tie1 = a + b
        item = (-w, -tie1, -a, -b, a, b)
        if item > heap[0]:
            heapq.heapreplace(heap, item)

    return [(item[4], item[5]) for item in heap]


def sort_by_slope(vs: List[Pair]) -> List[Pair]:
    """Sort by slope b/a (ascending). Float is safe for small integer pairs."""
    return sorted(vs, key=lambda p: (p[1] / p[0], p[0], p[1]))


def area_from_half_edges(half_edges: List[Tuple[int, int]]) -> int:
    """
    For a centrally symmetric polygon, the full area equals the sum of determinants
    over all ordered pairs in the half-cycle, which is equal to:
        sum_j det(prefix_sum_before_j, edge_j)
    This is O(m).
    """
    px = 0
    py = 0
    area = 0
    for dx, dy in half_edges:
        area += px * dy - py * dx
        px += dx
        py += dy
    return area


def polygon_area_from_interior(interior: List[Pair]) -> int:
    """
    Build the symmetric edge list half-cycle:
        (1,0) + interior_sorted + (0,1) + mirrored interior
    Then compute area via prefix determinant sum.
    """
    interior_sorted = sort_by_slope(interior)

    half = [(1, 0)]
    half.extend(interior_sorted)
    half.append((0, 1))
    half.extend([(-a, b) for (a, b) in reversed(interior_sorted)])

    return area_from_half_edges(half)


def compute_A(N: int) -> int:
    """
    Compute A(N): minimum area of a symmetrical convex grid polygon with N vertices.

    The reduction used in this solver implies:
        N = 4*(k+1)  =>  k = (N-4)/4  interior primitive vectors in the first quadrant.
    """
    if N < 4 or N % 4 != 0:
        raise ValueError(
            "This solver assumes N is a positive multiple of 4 (as in the problem)."
        )

    k = (N - 4) // 4
    if k == 0:
        return 1  # the minimal square

    # dynamically increase search limit if boundary points appear
    limit = 40
    pairs = primitive_pairs(limit)

    # scan t in [0.001, 1.000] at step 0.001
    best_area = None

    for tn in range(1, 1001):
        t = tn / 1000.0

        # Ensure we have enough candidate points; enlarge if needed.
        while True:
            chosen = n_smallest_by_weight(pairs, k, t)
            max_a = max(p[0] for p in chosen)
            max_b = max(p[1] for p in chosen)
            if max_a < limit and max_b < limit:
                break
            limit *= 2
            pairs = primitive_pairs(limit)

        area = polygon_area_from_interior(chosen)

        if best_area is None or area < best_area:
            best_area = area

    return best_area


def main() -> None:
    # Asserts from the problem statement:
    assert compute_A(4) == 1
    assert compute_A(8) == 7
    assert compute_A(40) == 1039
    assert compute_A(100) == 17473

    print(compute_A(1000))


if __name__ == "__main__":
    main()
