#!/usr/bin/env python3
"""Project Euler 842: Irregular Star Polygons

Idea
----
For a fixed n, consider *all* diagonals of the regular n-gon. Every interior
intersection point P is the meeting point of m>=2 diagonals, and every pair of
those diagonals corresponds to exactly one 4-tuple of vertices (a<b<c<d)
producing that intersection.

For a Hamiltonian cycle (our 'star polygon'), P is a self-intersection point iff
at least two of the m diagonals through P are used as edges of the cycle.
Because no diagonal sharing an endpoint can meet at an interior point, the m
edges through P are pairwise vertex-disjoint.

Therefore:
  T(n) = sum_over_intersection_points_P  count_cycles_using_at_least_2_of_m(P)

The geometric part reduces to counting, for the regular n-gon, how many distinct
intersection points have multiplicity m (number of diagonals through the point).
We get these multiplicities by enumerating all (a<b<c<d), computing the
intersection of diagonals (a,c) and (b,d), grouping equal points, and using the
fact that each point with multiplicity m appears exactly C(m,2) times.

The combinatorial part is purely graph-theoretic: in K_n, the number of
Hamiltonian cycles containing a fixed set of k disjoint edges equals
  A_k = 2^(k-1) * (n-k-1)!
(undirected cycles).
Using binomial inversion, we compute how many cycles contain at least two edges
from a given matching of size m.

We need sum_{n=3..60} T(n) modulo 1e9+7.

Constraints
-----------
No external libraries are used.

"""

from __future__ import annotations

import math
from math import isqrt

MOD = 1_000_000_007


def inv_triangular(q: int) -> int:
    """Given q = m*(m-1)//2, return m (m>=2)."""
    disc = 1 + 8 * q
    r = isqrt(disc)
    if r * r != disc or (1 + r) % 2 != 0:
        raise ValueError(f"count {q} is not triangular")
    return (1 + r) // 2


def cycles_at_least_two_edges(n: int, m: int, fact: list[int]) -> int:
    """Number of undirected Hamiltonian cycles on n vertices that use >=2 edges
    from a fixed matching of size m (m disjoint edges).

    Uses binomial inversion on:
      sum_{j>=k} C(j,k) * c_j = C(m,k) * A_k
    where c_j = #cycles using exactly j edges from the matching.
    """
    if m < 2:
        return 0
    total = fact[n - 1] // 2

    # A_0 = total, A_k = 2^(k-1) * (n-k-1)! for k>=1
    def A(k: int) -> int:
        if k == 0:
            return total
        return (1 << (k - 1)) * fact[n - k - 1]

    # c0 = cycles using none of the m edges
    c0 = 0
    for k in range(0, m + 1):
        term = math.comb(m, k) * A(k)
        c0 = c0 + term if (k % 2 == 0) else c0 - term

    # c1 = cycles using exactly one of the m edges
    c1 = 0
    for k in range(1, m + 1):
        term = k * math.comb(m, k) * A(k)
        c1 = c1 + term if ((k - 1) % 2 == 0) else c1 - term

    return total - c0 - c1


def multiplicity_distribution(n: int) -> dict[int, int]:
    """Return a dict {m: count_of_points_with_m_diagonals} for the regular n-gon.

    We enumerate all a<b<c<d and compute the intersection of diagonals (a,c)
    and (b,d). Each distinct intersection point is keyed by a quantized (x,y)
    pair. The stored value per key is q = number of times we saw it, which
    equals C(m,2). Then we recover m.

    NOTE: Uses floating-point geometry with aggressive quantization; the provided
    asserts (T(5), T(8)) guard correctness.
    """

    # Vertex coordinates on the unit circle.
    tau = 2.0 * math.pi
    xs = [math.cos(tau * k / n) for k in range(n)]
    ys = [math.sin(tau * k / n) for k in range(n)]

    # Fast path: direct quantization into a dictionary.
    # If the quantization happens to merge two distinct points, some group
    # counts will not be triangular; in that case we fall back to a slower but
    # safer spatial clustering routine.
    SCALE = 10**11  # 1e-11 resolution

    counts: dict[tuple[int, int], int] = {}

    # Enumerate all vertex quadruples.
    for a in range(n - 3):
        x1 = xs[a]
        y1 = ys[a]
        for b in range(a + 1, n - 2):
            x3 = xs[b]
            y3 = ys[b]
            abx = x3 - x1
            aby = y3 - y1
            for c in range(b + 1, n - 1):
                x2 = xs[c]
                y2 = ys[c]
                dx12 = x2 - x1
                dy12 = y2 - y1
                for d in range(c + 1, n):
                    x4 = xs[d]
                    y4 = ys[d]
                    dx34 = x4 - x3
                    dy34 = y4 - y3

                    denom = dx12 * dy34 - dy12 * dx34
                    # For convex position and crossing diagonals this should
                    # never be 0, but keep a tiny safeguard.
                    if abs(denom) < 1e-18:
                        continue

                    # t = cross((x3-x1,y3-y1), (dx34,dy34)) / cross((dx12,dy12), (dx34,dy34))
                    t = (abx * dy34 - aby * dx34) / denom
                    x = x1 + t * dx12
                    y = y1 + t * dy12

                    key = (int(round(x * SCALE)), int(round(y * SCALE)))
                    counts[key] = counts.get(key, 0) + 1

    # Try decoding via triangular numbers.
    dist: dict[int, int] = {}
    ok = True
    for q in counts.values():
        disc = 1 + 8 * q
        r = isqrt(disc)
        if r * r != disc:
            ok = False
            break

    if ok:
        for q in counts.values():
            m = inv_triangular(q)
            dist[m] = dist.get(m, 0) + 1
        return dist

    # Slow fallback: spatial hashing + epsilon merge.
    cell = 1e-6
    eps2 = (1e-9) ** 2
    grid: dict[tuple[int, int], list[int]] = {}
    reps_x: list[float] = []
    reps_y: list[float] = []
    reps_cnt: list[int] = []

    def add_point(x: float, y: float) -> None:
        ix = int(math.floor(x / cell))
        iy = int(math.floor(y / cell))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                key = (ix + dx, iy + dy)
                lst = grid.get(key)
                if not lst:
                    continue
                for idx in lst:
                    dx0 = x - reps_x[idx]
                    dy0 = y - reps_y[idx]
                    if dx0 * dx0 + dy0 * dy0 <= eps2:
                        reps_cnt[idx] += 1
                        return
        idx = len(reps_cnt)
        reps_x.append(x)
        reps_y.append(y)
        reps_cnt.append(1)
        grid.setdefault((ix, iy), []).append(idx)

    # Re-run the enumeration, but add points through clustering.
    for a in range(n - 3):
        x1 = xs[a]
        y1 = ys[a]
        for b in range(a + 1, n - 2):
            x3 = xs[b]
            y3 = ys[b]
            abx = x3 - x1
            aby = y3 - y1
            for c in range(b + 1, n - 1):
                x2 = xs[c]
                y2 = ys[c]
                dx12 = x2 - x1
                dy12 = y2 - y1
                for d in range(c + 1, n):
                    x4 = xs[d]
                    y4 = ys[d]
                    dx34 = x4 - x3
                    dy34 = y4 - y3
                    denom = dx12 * dy34 - dy12 * dx34
                    if abs(denom) < 1e-18:
                        continue
                    t = (abx * dy34 - aby * dx34) / denom
                    x = x1 + t * dx12
                    y = y1 + t * dy12
                    add_point(x, y)

    dist2: dict[int, int] = {}
    for q in reps_cnt:
        m = inv_triangular(q)
        dist2[m] = dist2.get(m, 0) + 1
    return dist2


def T(n: int, fact: list[int]) -> int:
    """Compute T(n) exactly as Python int."""
    if n < 4:
        return 0

    dist = multiplicity_distribution(n)
    # Precompute cycles count per multiplicity.
    g: dict[int, int] = {}
    for m in dist:
        g[m] = cycles_at_least_two_edges(n, m, fact)

    total = 0
    for m, cnt in dist.items():
        total += cnt * g[m]
    return total


def main() -> None:
    # Exact factorials up to 60.
    fact = [1] * 61
    for i in range(1, 61):
        fact[i] = fact[i - 1] * i

    # Problem statement checks.
    assert T(5, fact) == 20
    assert T(8, fact) == 14640

    ans_mod = 0
    for n in range(3, 61):
        ans_mod = (ans_mod + (T(n, fact) % MOD)) % MOD

    print(ans_mod)


if __name__ == "__main__":
    main()
