#!/usr/bin/env python3
"""
Project Euler 644: Squares on the Line

Compute f(200, 500) for the random first move game.
"""

from __future__ import annotations

from typing import List, Tuple
import math

SQRT2 = math.sqrt(2.0)
EPS = 1e-12


def _generate_ring(max_l: float) -> List[float]:
    vals = [0.0]
    max_b = int(max_l / SQRT2) + 1
    for b in range(max_b + 1):
        base = b * SQRT2
        max_a = int(max_l - base + 1e-12)
        for a in range(max_a + 1):
            vals.append(a + base)
    return sorted(set(vals))


def _compute_grundy_intervals(
    max_l: float,
) -> Tuple[List[float], List[float], List[int]]:
    vals = _generate_ring(max_l)
    n = len(vals) - 1
    starts: List[float] = []
    ends: List[float] = []
    grundy: List[int] = []
    moves = [False] * 256
    max_g = 0

    def upper_index(arr: List[float], x: float) -> int:
        lo = 0
        hi = len(arr)
        while lo < hi:
            mid = (lo + hi) // 2
            if arr[mid] <= x:
                lo = mid + 1
            else:
                hi = mid
        return lo

    for idx in range(n):
        a = vals[idx]
        b = vals[idx + 1]
        L = (a + b) * 0.5
        for i in range(max_g + 1):
            moves[i] = False

        for x in (1.0, SQRT2):
            if L < x or not starts:
                continue
            S = L - x
            u_idx = upper_index(starts, S) - 1
            t_idx = 0
            while t_idx <= u_idx:
                t_start = starts[t_idx]
                if t_start >= S:
                    break
                t_end = ends[t_idx]
                if t_end > S:
                    t_end = S
                u_start = starts[u_idx]
                u_end = ends[u_idx]
                if u_end > S:
                    u_end = S
                left = S - u_end
                if t_start > left:
                    left = t_start
                right = S - u_start
                if t_end < right:
                    right = t_end
                if left < right:
                    g = grundy[t_idx] ^ grundy[u_idx]
                    if g >= len(moves):
                        moves.extend([False] * len(moves))
                    moves[g] = True
                    if g > max_g:
                        max_g = g
                if t_end < S - u_start:
                    t_idx += 1
                else:
                    u_idx -= 1

        g = 0
        while g < len(moves) and moves[g]:
            g += 1
        if g >= len(moves):
            moves.extend([False] * len(moves))

        if grundy and grundy[-1] == g and abs(ends[-1] - a) < EPS:
            ends[-1] = b
        else:
            starts.append(a)
            ends.append(b)
            grundy.append(g)

    return starts, ends, grundy


def _build_w_segments(
    starts: List[float],
    ends: List[float],
    grundy: List[int],
    max_s: float,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    max_g = max(grundy) if grundy else 0
    groups: List[List[Tuple[float, float]]] = [[] for _ in range(max_g + 1)]
    for a, b, g in zip(starts, ends, grundy):
        groups[g].append((a, b))

    events: List[Tuple[float, int]] = []
    for intervals in groups:
        m = len(intervals)
        for i in range(m):
            a1, b1 = intervals[i]
            for j in range(i, m):
                a2, b2 = intervals[j]
                w = 1 if i == j else 2
                p0 = a1 + a2
                p1 = a1 + b2
                p2 = b1 + a2
                p3 = b1 + b2
                if p0 > max_s + EPS:
                    break
                if p3 < 0.0:
                    continue
                if p0 < 0.0:
                    p0 = 0.0
                if p3 > max_s:
                    p3 = max_s
                q1 = p1 if p1 < p2 else p2
                q2 = p2 if p1 < p2 else p1
                events.append((p0, w))
                events.append((q1, -w))
                events.append((q2, -w))
                events.append((p3, w))

    events.sort()
    merged: List[Tuple[float, int]] = []
    if events:
        cur_pos, cur_delta = events[0]
        for pos, delta in events[1:]:
            if abs(pos - cur_pos) < EPS:
                cur_delta += delta
            else:
                merged.append((cur_pos, cur_delta))
                cur_pos, cur_delta = pos, delta
        merged.append((cur_pos, cur_delta))

    seg_starts: List[float] = []
    seg_ends: List[float] = []
    seg_slopes: List[float] = []
    seg_vals: List[float] = []
    slope = 0.0
    val = 0.0
    prev = 0.0
    for pos, delta in merged:
        if pos > max_s:
            break
        if pos > prev:
            seg_starts.append(prev)
            seg_ends.append(pos)
            seg_slopes.append(slope)
            seg_vals.append(val)
            val += slope * (pos - prev)
            prev = pos
        slope += delta
    if prev < max_s:
        seg_starts.append(prev)
        seg_ends.append(max_s)
        seg_slopes.append(slope)
        seg_vals.append(val)
    return seg_starts, seg_ends, seg_slopes, seg_vals


def _w_value(
    seg_starts: List[float],
    seg_ends: List[float],
    seg_slopes: List[float],
    seg_vals: List[float],
    x: float,
) -> Tuple[float, float]:
    lo = 0
    hi = len(seg_starts)
    while lo < hi:
        mid = (lo + hi) // 2
        if seg_ends[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    if lo >= len(seg_starts):
        return 0.0, 0.0
    start = seg_starts[lo]
    slope = seg_slopes[lo]
    val = seg_vals[lo] + slope * (x - start)
    return val, slope


def _e_value(
    L: float, seg_data: Tuple[List[float], List[float], List[float], List[float]]
) -> float:
    seg_starts, seg_ends, seg_slopes, seg_vals = seg_data
    w1, _ = _w_value(seg_starts, seg_ends, seg_slopes, seg_vals, L - 1.0)
    w2, _ = _w_value(seg_starts, seg_ends, seg_slopes, seg_vals, L - SQRT2)
    return 0.5 * L * (w1 / (L - 1.0) + w2 / (L - SQRT2))


def _f_value(
    a: float,
    b: float,
    seg_data: Tuple[List[float], List[float], List[float], List[float]],
) -> Tuple[float, float]:
    seg_starts, seg_ends, seg_slopes, seg_vals = seg_data
    points = [a, b]
    for p in seg_starts:
        v = p + 1.0
        if a < v < b:
            points.append(v)
        v = p + SQRT2
        if a < v < b:
            points.append(v)
    for p in seg_ends:
        v = p + 1.0
        if a < v < b:
            points.append(v)
        v = p + SQRT2
        if a < v < b:
            points.append(v)
    points = sorted(set(points))

    best = -1.0
    bestL = a

    def term_deriv(m: float, b0: float, c: float, L: float) -> float:
        return (m * L * L - 2.0 * m * c * L - b0 * c) / ((L - c) * (L - c))

    for i in range(len(points) - 1):
        L0 = points[i]
        L1 = points[i + 1]
        if L1 - L0 < 1e-12:
            continue
        mid = (L0 + L1) * 0.5

        w1, m1 = _w_value(seg_starts, seg_ends, seg_slopes, seg_vals, mid - 1.0)
        w2, m2 = _w_value(seg_starts, seg_ends, seg_slopes, seg_vals, mid - SQRT2)
        b1 = w1 - m1 * mid
        b2 = w2 - m2 * mid

        def e_local(L: float) -> float:
            return 0.5 * L * ((m1 * L + b1) / (L - 1.0) + (m2 * L + b2) / (L - SQRT2))

        def de_local(L: float) -> float:
            return 0.5 * (term_deriv(m1, b1, 1.0, L) + term_deriv(m2, b2, SQRT2, L))

        for L in (L0, mid, L1):
            val = e_local(L)
            if val > best:
                best = val
                bestL = L

        left = L0 + 1e-10
        right = L1 - 1e-10
        if left >= right:
            continue
        dl = de_local(left)
        dm = de_local(mid)
        dr = de_local(right)

        def bisect(lo: float, hi: float, dlo: float, dhi: float) -> float:
            for _ in range(60):
                m = (lo + hi) * 0.5
                dm_local = de_local(m)
                if dm_local == 0.0:
                    return m
                if dm_local * dlo > 0.0:
                    lo = m
                    dlo = dm_local
                else:
                    hi = m
            return (lo + hi) * 0.5

        if dl * dm < 0.0:
            root = bisect(left, mid, dl, dm)
            val = e_local(root)
            if val > best:
                best = val
                bestL = root
        if dm * dr < 0.0:
            root = bisect(mid, right, dm, dr)
            val = e_local(root)
            if val > best:
                best = val
                bestL = root

    return best, bestL


def _round8(x: float) -> float:
    return float(f"{x:.8f}")


def main() -> None:
    max_l = 500.0
    starts, ends, grundy = _compute_grundy_intervals(max_l)
    seg_data = _build_w_segments(starts, ends, grundy, max_l)

    assert _round8(_e_value(2.0, seg_data)) == 2.0
    assert _round8(_e_value(4.0, seg_data)) == 1.11974851
    assert _round8(_f_value(2.0, 10.0, seg_data)[0]) == 2.61969775
    assert _round8(_f_value(10.0, 20.0, seg_data)[0]) == 5.99374121

    answer = _f_value(200.0, 500.0, seg_data)[0]
    print(f"{answer:.8f}")


if __name__ == "__main__":
    main()
