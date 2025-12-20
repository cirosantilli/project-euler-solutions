#!/usr/bin/env python3
"""
Project Euler 562 - Maximal perimeter (lattice triangles)

Constraints implemented:
- Standard library only (no external packages).
- Single-threaded.

Math used (rephrased):
- An "empty" lattice triangle (no other lattice points inside or on edges) must have area 1/2.
  This follows from Pick's theorem: Area = I + B/2 - 1, with I=0 and B=3.
- With A as origin, let u=B-A and v=C-A. Then Area = |det(u,v)|/2, so emptiness implies |det(u,v)|=1.
  This makes (u,v) a unimodular pair.
- For area 1/2, circumradius satisfies R = a*b*c / (4*Area) = a*b*c / 2.
  Therefore T(r)=R/r and T(r)^2 = (s1*s2*s3)/(4*r^2) where s1,s2,s3 are squared side lengths.

Search plan:
1) The longest side AB is extremely close to a diameter for the maximizing triangle.
   We only need to look at "outer" lattice points on the right boundary of the disk:
      p(y) = (x(y), y),  x(y) = max integer with x(y)^2 + y^2 <= r^2.
   Endpoints of a near-diameter chord can be chosen as B=p(i) and A=-p(j),
   giving base vector u = B-A = p(i)+p(j).

2) We filter candidate endpoints by how close they are to the circle:
      deficit(y) = r^2 - x(y)^2 - y^2
   and keep only those with deficit <= LIMIT (LIMIT=8000 works for r=10^7, and for small r it includes all).
   Among these, we find the longest primitive base vector u (gcd(u.x,u.y)=1).

3) For each best base candidate u, we construct the third vertex via extended gcd:
   Solve u.x*s + u.y*t = 1, then v0 = (-t, s) gives det(u,v0)=1.
   All solutions are v = ±v0 + k*u. We choose k so that C = A + v lies inside the circle
   (A is one of the two endpoint choices: -p(i) or -p(j)). Among feasible options we maximize the perimeter.

4) The final output is round(T(10^7)) using exact integer rounding on T(r)^2.
"""

from __future__ import annotations

import math
import sys
from typing import List, Tuple


LIMIT_DEFICIT = 8000  # near-boundary filter; fast and sufficient for the target input


def egcd(a: int, b: int) -> Tuple[int, int, int]:
    """Return (g,x,y) with a*x + b*y = g, for a,b >= 0."""
    x0, y0 = 1, 0
    x1, y1 = 0, 1
    while b:
        q = a // b
        a, b = b, a - q * b
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    return a, x0, y0


def ceil_div(a: int, b: int) -> int:
    """Ceiling division for integers with b > 0."""
    return -((-a) // b)


def boundary_points_near_circle(
    r: int, deficit_limit: int
) -> Tuple[List[int], List[int]]:
    """
    Produce lists xs, ys of points (x(y), y) on the right boundary of the disk
    with deficit <= deficit_limit.

    Uses a two-pointer walk: x only decreases as y increases, so total work is O(r).
    """
    r2 = r * r
    x = r
    x2 = x * x
    y2 = 0

    xs: List[int] = []
    ys: List[int] = []

    for y in range(r + 1):
        while x2 + y2 > r2:
            x -= 1
            x2 = x * x

        deficit = r2 - x2 - y2
        if deficit <= deficit_limit:
            xs.append(x)
            ys.append(y)

        # (y+1)^2 = y^2 + 2y + 1
        y2 += 2 * y + 1

    return xs, ys


def best_base_candidates(
    xs: List[int], ys: List[int]
) -> Tuple[int, List[Tuple[int, int, int, int]]]:
    """
    Find the maximum squared length u2 of u = p(i)+p(j) with i<j and gcd(u.x,u.y)=1.
    Returns (best_u2, list_of_candidates) where each candidate is (i,j,ux,uy).
    """
    m = len(xs)
    gcd = math.gcd

    best_u2 = -1
    best: List[Tuple[int, int, int, int]] = []

    for i in range(m):
        xi = xs[i]
        yi = ys[i]
        for j in range(i + 1, m):
            ux = xi + xs[j]
            uy = yi + ys[j]
            u2 = ux * ux + uy * uy

            if u2 < best_u2:
                continue

            # If both are even, gcd>1.
            if ((ux | uy) & 1) == 0:
                continue

            if gcd(ux, uy) != 1:
                continue

            if u2 > best_u2:
                best_u2 = u2
                best = [(i, j, ux, uy)]
            elif u2 == best_u2:
                best.append((i, j, ux, uy))

    if best_u2 < 0:
        raise RuntimeError("No primitive base found; increase deficit limit.")
    return best_u2, best


def best_triangle_for_base(
    r: int, ax: int, ay: int, ux: int, uy: int
) -> Tuple[int, int, int, float]:
    """
    Given A=(ax,ay) and base vector u=(ux,uy) (B=A+u), find C=A+v with det(u,v)=±1
    and all vertices inside the circle.

    Returns (s1,s2,s3,perimeter) where s1=|u|^2, s2=|v|^2, s3=|u-v|^2.
    Raises if no feasible C exists.
    """
    r2 = r * r

    # Quick feasibility for A,B.
    if ax * ax + ay * ay > r2:
        raise RuntimeError("A outside circle (unexpected).")
    bx = ax + ux
    by = ay + uy
    if bx * bx + by * by > r2:
        raise RuntimeError("B outside circle (unexpected).")

    s1 = ux * ux + uy * uy

    g, s, t = egcd(abs(ux), abs(uy))
    if g != 1:
        raise RuntimeError("Base vector not primitive (unexpected).")
    if ux < 0:
        s = -s
    if uy < 0:
        t = -t

    # v0 gives det(u, v0) = 1
    v0x, v0y = -t, s

    best_sides = None
    best_per = -1.0

    isqrt = math.isqrt
    sqrt = math.sqrt

    for sign in (1, -1):  # det = +1 or -1
        bvx, bvy = sign * v0x, sign * v0y

        # v = b + k*u, C = A + v must lie in the circle.
        p0x = ax + bvx
        p0y = ay + bvy
        pu = p0x * ux + p0y * uy
        pp = p0x * p0x + p0y * p0y

        # Quadratic in k: |p0 + k*u|^2 <= r^2
        uu = s1
        disc = pu * pu - uu * (pp - r2)
        if disc < 0:
            continue
        root = isqrt(disc)

        k_lo = ceil_div(-pu - root, uu)
        k_hi = (-pu + root) // uu

        for k in range(k_lo, k_hi + 1):
            vx = bvx + k * ux
            vy = bvy + k * uy
            cx = ax + vx
            cy = ay + vy
            if cx * cx + cy * cy > r2:
                continue

            s2 = vx * vx + vy * vy
            wx = ux - vx
            wy = uy - vy
            s3 = wx * wx + wy * wy

            per = sqrt(s1) + sqrt(s2) + sqrt(s3)
            if per > best_per:
                best_per = per
                best_sides = (s1, s2, s3)

    if best_sides is None:
        raise RuntimeError("No feasible third vertex for this base.")
    return best_sides[0], best_sides[1], best_sides[2], best_per


def round_sqrt_rational(numer: int, den: int) -> int:
    """
    Return nearest integer to sqrt(numer/den), numer,den > 0.
    Uses exact integer comparisons.
    """
    k = math.isqrt(numer // den)
    t = 2 * k + 1
    # Round up iff numer/den >= (2k+1)^2 / 4  <=>  4*numer >= den*(2k+1)^2
    if 4 * numer >= den * t * t:
        return k + 1
    return k


def compute_T_value(r: int, deficit_limit: int = LIMIT_DEFICIT) -> int:
    """
    Compute round(T(r)) for the disk radius r.
    """
    xs, ys = boundary_points_near_circle(r, deficit_limit)
    _, base_cands = best_base_candidates(xs, ys)

    best_sides = None
    best_per = -1.0

    for i, j, ux, uy in base_cands:
        # Two symmetric endpoint choices:
        # B = p(i), A = -p(j)  OR  B = p(j), A = -p(i)
        for ax, ay in [(-xs[i], -ys[i]), (-xs[j], -ys[j])]:
            try:
                s1, s2, s3, per = best_triangle_for_base(r, ax, ay, ux, uy)
            except RuntimeError:
                continue
            if per > best_per:
                best_per = per
                best_sides = (s1, s2, s3)

    if best_sides is None:
        raise RuntimeError("No triangle found; increase deficit limit.")

    s1, s2, s3 = best_sides
    numer = s1 * s2 * s3
    den = 4 * r * r
    return round_sqrt_rational(numer, den)


def _self_test() -> None:
    # Statement-provided checks:
    #
    # For r=5, T(5)^2 = 19669/50.
    # Since T(r)^2 = (s1*s2*s3)/(4*r^2) and 4*r^2=100, we must have s1*s2*s3 = 39338.
    #
    # Also given approximate values for r=10 and r=100.
    def best_product(r: int) -> int:
        xs, ys = boundary_points_near_circle(r, LIMIT_DEFICIT)
        _, base_cands = best_base_candidates(xs, ys)
        best = 0
        for i, j, ux, uy in base_cands:
            for ax, ay in [(-xs[i], -ys[i]), (-xs[j], -ys[j])]:
                try:
                    s1, s2, s3, _ = best_triangle_for_base(r, ax, ay, ux, uy)
                except RuntimeError:
                    continue
                prod = s1 * s2 * s3
                if prod > best:
                    best = prod
        return best

    prod5 = best_product(5)
    assert prod5 == 39338, prod5

    # Approximate values in the statement
    t10 = math.sqrt(best_product(10) / (4.0 * 10 * 10))
    assert abs(t10 - 97.26729) < 1e-5, t10

    t100 = math.sqrt(best_product(100) / (4.0 * 100 * 100))
    assert abs(t100 - 9157.64707) < 1e-5, t100


def main() -> None:
    _self_test()

    r = 10**7
    if len(sys.argv) >= 2:
        r = int(sys.argv[1])

    print(compute_T_value(r))


if __name__ == "__main__":
    main()
