#!/usr/bin/env python3
"""
Project Euler 727 - Triangle of Circular Arcs

Computes the expected distance d = |DE| between:
- D: circumcenter of the triangle formed by the three tangency points of three
     mutually externally tangent circles with radii ra < rb < rc.
- E: center of the inner Soddy circle (the circle externally tangent to all three).

No external libraries are used (stdlib only).
"""

from __future__ import annotations

import math
import sys


def inner_soddy_radius(ra: int, rb: int, rc: int) -> float:
    """Radius of the inner Soddy circle tangent to three mutually externally tangent circles."""
    k1 = 1.0 / ra
    k2 = 1.0 / rb
    k3 = 1.0 / rc
    k4 = k1 + k2 + k3 + 2.0 * math.sqrt(k1 * k2 + k2 * k3 + k3 * k1)  # inner solution
    return 1.0 / k4


def _safe_sqrt(x: float) -> float:
    """sqrt with tiny negative clamped to 0 (rounding noise)."""
    if x < 0.0 and x > -1e-12:
        x = 0.0
    return math.sqrt(x)


def circle_C_center(ra: int, rb: int, rc: int) -> tuple[float, float]:
    """
    Place circle A at (0,0), circle B at (ra+rb, 0).
    Return coordinates of circle C (with positive y).
    """
    d_ab = ra + rb
    ac = ra + rc
    bc = rb + rc

    xC = (ac * ac - bc * bc + d_ab * d_ab) / (2.0 * d_ab)
    yC = _safe_sqrt(ac * ac - xC * xC)
    return xC, yC


def tangency_points(
    ra: int, rb: int, rc: int, xC: float, yC: float
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """
    Return tangency points (Tab, Tac, Tbc) between circles (A,B), (A,C), (B,C)
    in the coordinate system used by circle_C_center().
    """
    d_ab = ra + rb
    ac = ra + rc
    bc = rb + rc

    # A=(0,0), B=(d_ab,0)
    Tab = (float(ra), 0.0)

    # along AC from A by distance ra
    Tac = (ra * xC / ac, ra * yC / ac)

    # along BC from B by distance rb
    Tbc = (d_ab + rb * (xC - d_ab) / bc, rb * yC / bc)
    return Tab, Tac, Tbc


def circumcenter(
    p1: tuple[float, float], p2: tuple[float, float], p3: tuple[float, float]
) -> tuple[float, float]:
    """Circumcenter of the (non-collinear) triangle (p1,p2,p3)."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    det = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    # det should never be 0 for valid inputs
    if det == 0.0:
        raise ValueError("Degenerate triangle while computing circumcenter")

    s1 = x1 * x1 + y1 * y1
    s2 = x2 * x2 + y2 * y2
    s3 = x3 * x3 + y3 * y3

    ux = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / det
    uy = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / det
    return ux, uy


def incircle_center_E(
    ra: int, rb: int, rc: int, xC: float, yC: float, r4: float
) -> tuple[float, float]:
    """
    Center E of the inner Soddy circle.

    Uses intersection of circles around A and B with radii (ra+r4) and (rb+r4).
    Picks the intersection on the same side of AB as C (positive y in our placement).
    """
    d_ab = ra + rb
    RA = ra + r4
    RB = rb + r4

    # A=(0,0), B=(d_ab,0): circle-circle intersection has closed form
    xE = (RA * RA - RB * RB + d_ab * d_ab) / (2.0 * d_ab)
    yE = _safe_sqrt(RA * RA - xE * xE)

    # Ensure we pick the intersection that also matches tangency to C.
    # Typically yE>0 works; fall back to -yE if needed due to rounding.
    target = rc + r4
    dist_pos = math.hypot(xE - xC, yE - yC)
    if abs(dist_pos - target) < 1e-8:
        return xE, yE
    dist_neg = math.hypot(xE - xC, -yE - yC)
    if abs(dist_neg - target) < 1e-8:
        return xE, -yE

    # If both are slightly off, prefer same side as C.
    return xE, yE


def distance_DE(ra: int, rb: int, rc: int) -> float:
    """Compute d=|DE| for a given (ra,rb,rc)."""
    xC, yC = circle_C_center(ra, rb, rc)
    Tab, Tac, Tbc = tangency_points(ra, rb, rc, xC, yC)
    Dx, Dy = circumcenter(Tab, Tac, Tbc)

    r4 = inner_soddy_radius(ra, rb, rc)
    Ex, Ey = incircle_center_E(ra, rb, rc, xC, yC, r4)

    return math.hypot(Dx - Ex, Dy - Ey)


def expected_distance(limit: int = 100) -> float:
    """Expected value of d over integer triples 1<=ra<rb<rc<=limit with gcd(ra,rb,rc)=1."""
    count = 0
    total = 0.0
    comp = 0.0  # Kahan compensation

    for ra in range(1, limit + 1):
        for rb in range(ra + 1, limit + 1):
            g_ab = math.gcd(ra, rb)
            for rc in range(rb + 1, limit + 1):
                if math.gcd(g_ab, rc) != 1:
                    continue
                count += 1
                d = distance_DE(ra, rb, rc)

                # Kahan summation
                y = d - comp
                t = total + y
                comp = (t - total) - y
                total = t

    if count == 0:
        raise ValueError("No valid triples found")
    return total / count


def _self_test() -> None:
    # Basic geometric sanity checks on a small configuration.
    ra, rb, rc = 1, 2, 3
    xC, yC = circle_C_center(ra, rb, rc)
    Tab, Tac, Tbc = tangency_points(ra, rb, rc, xC, yC)
    Dx, Dy = circumcenter(Tab, Tac, Tbc)
    rD1 = math.hypot(Dx - Tab[0], Dy - Tab[1])
    rD2 = math.hypot(Dx - Tac[0], Dy - Tac[1])
    rD3 = math.hypot(Dx - Tbc[0], Dy - Tbc[1])
    assert max(rD1, rD2, rD3) - min(rD1, rD2, rD3) < 1e-10

    r4 = inner_soddy_radius(ra, rb, rc)
    Ex, Ey = incircle_center_E(ra, rb, rc, xC, yC, r4)
    assert abs(math.hypot(Ex - 0.0, Ey - 0.0) - (ra + r4)) < 1e-8
    assert abs(math.hypot(Ex - (ra + rb), Ey - 0.0) - (rb + r4)) < 1e-8
    assert abs(math.hypot(Ex - xC, Ey - yC) - (rc + r4)) < 1e-8


def main() -> None:
    _self_test()

    limit = 100
    if len(sys.argv) >= 2:
        limit = int(sys.argv[1])

    ans = expected_distance(limit)
    print(f"{ans:.8f}")


if __name__ == "__main__":
    main()
