from __future__ import annotations

import itertools
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


def find_min_radius_sq_for_parity_family(k: int, m_limit: int, filtered: bool) -> int:
    """
    Search the parity-subset construction with k vectors:
    centers = all even subset sums, so number of circles is 2^(k-1).
    """
    masks = _even_masks(k)
    n = 1 << (k - 1)

    for m in range(1, m_limit + 1):
        circle_points = lattice_points_on_circle(m)
        p = len(circle_points) // 2
        if p < k:
            continue

        # Filters from the previous verified rollout: they keep the k=10 search practical.
        if filtered:
            if p not in (k, k + 2):
                continue
            if not _has_unit_coordinate(circle_points):
                continue

        pairs = opposite_pairs(circle_points)
        if len(pairs) != p:
            continue

        for comb in itertools.combinations(range(p), k):
            chosen_pairs = [pairs[i] for i in comb]

            # Global sign flip leaves the construction equivalent.
            for bits in range(1 << (k - 1)):
                oriented: List[Point] = [chosen_pairs[0][0]]
                for i in range(1, k):
                    oriented.append(chosen_pairs[i][(bits >> (i - 1)) & 1])

                centers = _centers_from_vectors(oriented, masks)
                if not _quick_harmony_count_equals_n(centers, circle_points, n):
                    continue
                if _strict_perfect_check(centers, circle_points, n):
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
