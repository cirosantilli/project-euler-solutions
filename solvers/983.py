#!/usr/bin/env python
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

Point = Tuple[int, int]

# Packed-coordinate encoding for faster dict/set operations.
_OFF = 1 << 15
_SHIFT = 17


def _encode(x: int, y: int) -> int:
    return ((x + _OFF) << _SHIFT) | (y + _OFF)


def lattice_points_on_circle(m: int) -> List[Point]:
    lim = math.isqrt(m)
    points: List[Point] = []
    for x in range(-lim, lim + 1):
        y2 = m - x * x
        if y2 < 0:
            continue
        y = math.isqrt(y2)
        if y * y == y2:
            points.append((x, y))
            if y:
                points.append((x, -y))
    return points


def opposite_pairs(points: Sequence[Point]) -> List[Tuple[Point, Point]]:
    pairs: List[Tuple[Point, Point]] = []
    used = set()
    for v in sorted(points):
        if v in used:
            continue
        w = (-v[0], -v[1])
        used.add(v)
        used.add(w)
        pairs.append((v, w))
    return pairs


def antipodal_pair_count(m: int) -> int:
    """
    Count antipodal lattice-point pairs on x^2 + y^2 = m.

    By the two-squares theorem, primes 3 mod 4 must occur to even powers, and
    each prime 1 mod 4 with exponent a contributes a factor a + 1 to the
    number of lattice points.
    """
    x = m
    while x % 2 == 0:
        x //= 2

    product = 1
    p = 3
    while p * p <= x:
        if x % p == 0:
            exponent = 0
            while x % p == 0:
                x //= p
                exponent += 1
            if p % 4 == 1:
                product *= exponent + 1
            elif exponent & 1:
                return 0
        p += 2

    if x > 1:
        if x % 4 == 1:
            product *= 2
        elif x % 4 == 3:
            return 0

    return 2 * product


def _displacement_set(points: Sequence[Point]) -> set[int]:
    deltas: set[int] = set()
    for ax, ay in points:
        for bx, by in points:
            if ax != bx or ay != by:
                deltas.add(_encode(ax - bx, ay - by))
    return deltas


def _passes_four_vector_prune(
    selected: Sequence[Point], candidate: Point, deltas: set[int]
) -> bool:
    """
    Reject a branch if a signed sum of four selected directions is already a
    displacement between two radius vectors.
    """
    if len(selected) < 3:
        return True

    vx, vy = candidate
    selected_len = len(selected)
    for i in range(selected_len - 2):
        ax, ay = selected[i]
        for j in range(i + 1, selected_len - 1):
            bx, by = selected[j]
            for k in range(j + 1, selected_len):
                cx, cy = selected[k]
                for sa in (1, -1):
                    x1 = vx + sa * ax
                    y1 = vy + sa * ay
                    for sb in (1, -1):
                        x2 = x1 + sb * bx
                        y2 = y1 + sb * by

                        x = x2 + cx
                        y = y2 + cy
                        if (x or y) and _encode(x, y) in deltas:
                            return False

                        x = x2 - cx
                        y = y2 - cy
                        if (x or y) and _encode(x, y) in deltas:
                            return False

    return True


def _even_masks(k: int) -> List[int]:
    return [mask for mask in range(1 << k) if (mask.bit_count() & 1) == 0]


def _centers_from_vectors(
    vectors: Sequence[Point], masks: Sequence[int]
) -> List[Point]:
    centers: List[Point] = []
    for mask in masks:
        x = 0
        y = 0
        mm = mask
        while mm:
            lsb = mm & -mm
            idx = lsb.bit_length() - 1
            vx, vy = vectors[idx]
            x += vx
            y += vy
            mm -= lsb
        centers.append((x, y))
    return centers


def _quick_harmony_count_equals_n(
    centers: Sequence[Point], circle_points: Sequence[Point], n: int
) -> bool:
    # Count only points touched by at least two circles; stop early if too many.
    counts: Dict[int, int] = {}
    harmony_count = 0
    for cx, cy in centers:
        for vx, vy in circle_points:
            key = _encode(cx + vx, cy + vy)
            cur = counts.get(key)
            if cur is None:
                counts[key] = 1
            elif cur == 1:
                counts[key] = 2
                harmony_count += 1
                if harmony_count > n:
                    return False
            else:
                counts[key] = cur + 1
    return harmony_count == n


def _strict_perfect_check(
    centers: Sequence[Point], circle_points: Sequence[Point], n: int
) -> bool:
    center_codes = [_encode(x, y) for x, y in centers]
    center_set = set(center_codes)

    # Requirement 4: no tangent circle pairs.
    tangent_diffs = [_encode(2 * x, 2 * y) for x, y in circle_points]
    for c in center_codes:
        for d in tangent_diffs:
            other = c + d
            if other in center_set and c < other:
                return False

    # Build harmony-point incidences and connected components together.
    point_to_centers: Dict[int, List[int]] = {}
    for idx, (cx, cy) in enumerate(centers):
        for vx, vy in circle_points:
            key = _encode(cx + vx, cy + vy)
            arr = point_to_centers.get(key)
            if arr is None:
                point_to_centers[key] = [idx]
            else:
                arr.append(idx)

    harmony_points = [k for k, v in point_to_centers.items() if len(v) >= 2]
    if len(harmony_points) != n:
        return False

    parent = list(range(n))
    size = [1] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    for key in harmony_points:
        lst = point_to_centers[key]
        base = lst[0]
        for j in range(1, len(lst)):
            union(base, lst[j])

    root = find(0)
    for i in range(1, n):
        if find(i) != root:
            return False
    return True


def _has_unit_coordinate(points: Sequence[Point]) -> bool:
    for x, y in points:
        if abs(x) == 1 or abs(y) == 1:
            return True
    return False


def _has_valid_oriented_vectors(
    pairs: Sequence[Tuple[Point, Point]],
    circle_points: Sequence[Point],
    k: int,
    masks: Sequence[int],
    n: int,
) -> bool:
    deltas = _displacement_set(circle_points)
    selected: List[Point] = []

    def dfs(start: int) -> bool:
        if len(selected) == k:
            centers = _centers_from_vectors(selected, masks)
            return _quick_harmony_count_equals_n(
                centers, circle_points, n
            ) and _strict_perfect_check(centers, circle_points, n)

        needed = k - len(selected)
        for pair_index in range(start, len(pairs) - needed + 1):
            # A global sign flip leaves the construction equivalent, so the
            # first chosen pair needs only one orientation.
            choices = (pairs[pair_index][0],) if not selected else pairs[pair_index]
            for vector in choices:
                if not _passes_four_vector_prune(selected, vector, deltas):
                    continue
                selected.append(vector)
                if dfs(pair_index + 1):
                    return True
                selected.pop()
        return False

    return dfs(0)


def find_min_radius_sq_for_parity_family(k: int, m_limit: int, filtered: bool) -> int:
    """
    Search the parity-subset construction with k vectors:
    centers = all even subset sums, so number of circles is 2^(k-1).
    """
    masks = _even_masks(k)
    n = 1 << (k - 1)

    for m in range(1, m_limit + 1):
        p = antipodal_pair_count(m)
        if p < k:
            continue

        # Filters from the previous verified rollout: they keep the k=10 search practical.
        if filtered and p not in (k, k + 2):
            continue

        circle_points = lattice_points_on_circle(m)

        if filtered:
            if not _has_unit_coordinate(circle_points):
                continue

        pairs = opposite_pairs(circle_points)
        if len(pairs) != p:
            continue

        if _has_valid_oriented_vectors(pairs, circle_points, k, masks, n):
            return m

    raise RuntimeError(f"No solution found up to m={m_limit}")


def main() -> None:
    # Problem statement checks.
    assert find_min_radius_sq_for_parity_family(k=2, m_limit=20, filtered=False) == 1
    assert find_min_radius_sq_for_parity_family(k=3, m_limit=50, filtered=False) == 5

    # Need at least 500 circles, and 2^(10-1) = 512.
    print(find_min_radius_sq_for_parity_family(k=10, m_limit=20000, filtered=True))


if __name__ == "__main__":
    main()
