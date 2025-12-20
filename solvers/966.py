#!/usr/bin/env python3
"""
Project Euler 966: Triangle Circle Intersection

Let I(a,b,c) be the largest possible area of intersection between:
- a triangle with side lengths a,b,c, and
- a circle with the same area as the triangle,
allowing the circle to be translated freely.

Compute:
  sum I(a,b,c) for integers 1 <= a <= b <= c < a+b and a+b+c <= 200

No external libraries are used.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

PI = math.acos(-1.0)

# Numerical tolerances
_EPS = 1e-12


# -------------------------
# Segment-circle intersections (circle centered at origin)
# -------------------------


def _segment_circle_ts(
    ax: float, ay: float, bx: float, by: float, r2: float
) -> Tuple[int, float, float]:
    """
    Intersections of segment A->B with circle centered at origin, radius^2 = r2.

    Returns (k, t1, t2) where k in {0,1,2} and the t values are in [0,1].
    If k==1, t1 is the only intersection parameter.
    """
    dx = bx - ax
    dy = by - ay
    qa = dx * dx + dy * dy
    if qa < 1e-18:
        return (0, 0.0, 0.0)

    qb = 2.0 * (ax * dx + ay * dy)
    qc = ax * ax + ay * ay - r2

    disc = qb * qb - 4.0 * qa * qc
    if disc < -1e-12:
        return (0, 0.0, 0.0)
    if disc < 0.0:
        disc = 0.0

    sdisc = math.sqrt(disc)
    inv2a = 0.5 / qa
    t1 = (-qb - sdisc) * inv2a
    t2 = (-qb + sdisc) * inv2a

    ts = []
    if -_EPS <= t1 <= 1.0 + _EPS:
        ts.append(0.0 if t1 < 0.0 else (1.0 if t1 > 1.0 else t1))
    if -_EPS <= t2 <= 1.0 + _EPS:
        t2c = 0.0 if t2 < 0.0 else (1.0 if t2 > 1.0 else t2)
        # avoid duplicating tangency
        if not ts or abs(t2c - ts[0]) > 1e-11:
            ts.append(t2c)

    if not ts:
        return (0, 0.0, 0.0)
    if len(ts) == 1:
        return (1, ts[0], 0.0)

    ts.sort()
    return (2, ts[0], ts[1])


# -------------------------
# Triangle–circle intersection area (circle at origin)
#
# Key fix vs the earlier version:
#   - after splitting edges at circle intersections, classify each subsegment
#     using its endpoints (inside/ outside), not midpoint
#   - project computed intersection points back onto the circle to remove drift
# -------------------------


def _tri_or_sector(px: float, py: float, qx: float, qy: float, r2: float) -> float:
    """
    Signed area contribution for triangle (0,p,q) intersected with circle.
    If both endpoints are inside/on circle => triangle area; else sector area.
    """
    cross = px * qy - py * qx
    p2 = px * px + py * py
    q2 = qx * qx + qy * qy
    if p2 <= r2 + 1e-12 and q2 <= r2 + 1e-12:
        return 0.5 * cross
    dot = px * qx + py * qy
    ang = math.atan2(cross, dot)
    return 0.5 * r2 * ang


def _project_to_circle(x: float, y: float, r: float) -> Tuple[float, float]:
    d = math.hypot(x, y)
    if d < 1e-18:
        return x, y
    s = r / d
    return x * s, y * s


def _edge_contrib(ax: float, ay: float, bx: float, by: float, r2: float) -> float:
    """
    Signed contribution of an oriented polygon edge A->B to polygon∩circle area
    using the standard per-edge triangle/sector decomposition.
    """
    k, t1, t2 = _segment_circle_ts(ax, ay, bx, by, r2)
    if k == 0:
        return _tri_or_sector(ax, ay, bx, by, r2)

    dx = bx - ax
    dy = by - ay
    r = math.sqrt(r2)

    if k == 1:
        ix = ax + dx * t1
        iy = ay + dy * t1
        # project intersection back onto circle
        ix, iy = _project_to_circle(ix, iy, r)
        return _tri_or_sector(ax, ay, ix, iy, r2) + _tri_or_sector(ix, iy, bx, by, r2)

    # k == 2
    i1x = ax + dx * t1
    i1y = ay + dy * t1
    i2x = ax + dx * t2
    i2y = ay + dy * t2
    i1x, i1y = _project_to_circle(i1x, i1y, r)
    i2x, i2y = _project_to_circle(i2x, i2y, r)

    return (
        _tri_or_sector(ax, ay, i1x, i1y, r2)
        + _tri_or_sector(i1x, i1y, i2x, i2y, r2)
        + _tri_or_sector(i2x, i2y, bx, by, r2)
    )


def triangle_circle_intersection_area(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
    ox: float,
    oy: float,
    r: float,
) -> float:
    """Area of intersection between triangle ABC and circle centered at O with radius r."""
    r2 = r * r
    a1x, a1y = ax - ox, ay - oy
    b1x, b1y = bx - ox, by - oy
    c1x, c1y = cx - ox, cy - oy

    total = 0.0
    total += _edge_contrib(a1x, a1y, b1x, b1y, r2)
    total += _edge_contrib(b1x, b1y, c1x, c1y, r2)
    total += _edge_contrib(c1x, c1y, a1x, a1y, r2)
    return abs(total)


# -------------------------
# Triangle construction from side lengths
# -------------------------


def triangle_coords_from_sides(
    a: int, b: int, c: int
) -> Tuple[float, float, float, float, float, float]:
    """
    Place a triangle with side lengths a,b,c in the plane:

      A = (0,0)
      B = (c,0)
      C = (x,y) such that |AC| = b and |BC| = a

    (So side AB has length c.)
    """
    ax, ay = 0.0, 0.0
    bx, by = float(c), 0.0

    x = (b * b + c * c - a * a) / (2.0 * c)
    y2 = b * b - x * x
    if y2 < 0.0 and y2 > -1e-12:
        y2 = 0.0
    cy = math.sqrt(y2) if y2 > 0.0 else 0.0
    cx = x
    return ax, ay, bx, by, cx, cy


def triangle_area(
    ax: float, ay: float, bx: float, by: float, cx: float, cy: float
) -> float:
    return abs(0.5 * ((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)))


def circumcenter(
    ax: float, ay: float, bx: float, by: float, cx: float, cy: float
) -> Tuple[bool, float, float]:
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-15:
        return (False, 0.0, 0.0)
    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return (True, ux, uy)


# -------------------------
# Maximization (deterministic pattern search)
# -------------------------


def maximize_intersection_for_triangle(a: int, b: int, c: int) -> float:
    """Compute I(a,b,c)."""
    ax, ay, bx, by, cx, cy = triangle_coords_from_sides(a, b, c)
    area = triangle_area(ax, ay, bx, by, cx, cy)
    if area <= 0.0:
        return 0.0

    r = math.sqrt(area / PI)

    xmin = min(ax, bx, cx) - r
    xmax = max(ax, bx, cx) + r
    ymin = min(ay, by, cy) - r
    ymax = max(ay, by, cy) + r

    cenx = (ax + bx + cx) / 3.0
    ceny = (ay + by + cy) / 3.0

    per = a + b + c
    incx = (a * ax + b * bx + c * cx) / per
    incy = (a * ay + b * by + c * cy) / per

    has_cc, ccx, ccy = circumcenter(ax, ay, bx, by, cx, cy)

    starts = [
        (cenx, ceny),
        (incx, incy),
        (ax, ay),
        (bx, by),
        (cx, cy),
        ((ax + bx) * 0.5, (ay + by) * 0.5),
        ((bx + cx) * 0.5, (by + cy) * 0.5),
        ((cx + ax) * 0.5, (cy + ay) * 0.5),
    ]
    if has_cc:
        starts.append((ccx, ccy))

    def clamp(px: float, py: float) -> Tuple[float, float]:
        if px < xmin:
            px = xmin
        elif px > xmax:
            px = xmax
        if py < ymin:
            py = ymin
        elif py > ymax:
            py = ymax
        return px, py

    def eval_at(px: float, py: float) -> float:
        v = triangle_circle_intersection_area(ax, ay, bx, by, cx, cy, px, py, r)
        # guard against tiny numerical overshoots
        if v < 0.0:
            return 0.0
        if v > area:
            return area
        return v

    bestx, besty = clamp(cenx, ceny)
    bestv = eval_at(bestx, besty)
    for sx, sy in starts:
        sx, sy = clamp(sx, sy)
        v = eval_at(sx, sy)
        if v > bestv:
            bestv, bestx, besty = v, sx, sy

    span = max(xmax - xmin, ymax - ymin)
    step = max(span, r)

    # 8 directions to avoid axis-aligned stall near ridges
    dirs = (
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (1.0, 1.0),
        (1.0, -1.0),
        (-1.0, 1.0),
        (-1.0, -1.0),
    )

    # stopping threshold (scale-aware)
    tol_step = max(1e-7, 1e-7 * max(1.0, r))

    while step > tol_step:
        moved_any = False

        # keep walking at the current step as long as we improve
        while True:
            improved = False
            bx0, by0, bv0 = bestx, besty, bestv
            for dx, dy in dirs:
                px, py = clamp(bestx + dx * step, besty + dy * step)
                v = eval_at(px, py)
                if v > bv0 + 1e-13:
                    bv0, bx0, by0 = v, px, py
                    improved = True
            if improved:
                bestv, bestx, besty = bv0, bx0, by0
                moved_any = True
            else:
                break

        if not moved_any:
            step *= 0.5

    return bestv


# -------------------------
# Enumerate triangles and solve
# -------------------------


def build_shape_coeffs(limit: int) -> Dict[Tuple[int, int, int], int]:
    """
    Group triangles by gcd-reduced (primitive) shape.
    For each primitive shape (a',b',c'), accumulate sum of scale^2.

    If (a,b,c) = g*(a',b',c'), then I(a,b,c) = g^2 * I(a',b',c').
    """
    coeff: Dict[Tuple[int, int, int], int] = {}
    for a in range(1, limit + 1):
        for b in range(a, limit + 1):
            maxc = min(a + b - 1, limit - a - b)
            if maxc < b:
                continue
            for c in range(b, maxc + 1):
                g = math.gcd(a, math.gcd(b, c))
                key = (a // g, b // g, c // g)
                coeff[key] = coeff.get(key, 0) + g * g
    return coeff


def solve(limit: int = 200) -> float:
    coeff = build_shape_coeffs(limit)

    # Kahan summation for stability
    total = 0.0
    corr = 0.0

    for (a, b, c), w in coeff.items():
        i0 = maximize_intersection_for_triangle(a, b, c)
        term = i0 * w
        y = term - corr
        t = total + y
        corr = (t - total) - y
        total = t

    return total


def _self_test() -> None:
    # Test values given in the problem statement
    v1 = maximize_intersection_for_triangle(3, 4, 5)
    v2 = maximize_intersection_for_triangle(3, 4, 6)
    assert abs(v1 - 4.593049) < 2e-6
    assert abs(v2 - 3.552564) < 2e-6


if __name__ == "__main__":
    _self_test()
    ans = solve(200)
    print(f"{ans:.2f}")
