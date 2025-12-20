#!/usr/bin/env python3
"""Project Euler 894: Spiral of Circles.

C_k is produced from C_{k-1} by the same scaling (factor s, 0 < s < 1) and
rotation (angle theta) about the origin. Therefore radius(C_k) = s^k and centres
follow a logarithmic spiral.

Given that C0 is externally tangent to C1, C7 and C8 and circles do not overlap,
we want the total area of all "circular triangles" (curvilinear triangles
bounded by three circular arcs between three mutually tangent circles).

Geometry and self-similarity
----------------------------
Let z = s * e^{i*theta}. If the distance from the origin to the centre of C0 is
d, then the centre of C_n is z^n times the centre of C0. External tangency
between C0 (radius 1) and C_n (radius s^n) gives:

  d * |1 - z^n| = 1 + s^n.

For n = 1, 7, 8 the left-hand side shares the same d, so

  |1 - z|/(1+s) = |1 - z^7|/(1+s^7) = |1 - z^8|/(1+s^8).

We solve two equations (n=7 and n=8) for (s, theta).

If C0 is tangent to C1, C7 and C8, then by applying the same similarity
transform repeatedly, every C_k is tangent to C_{k+1}, C_{k+7}, and C_{k+8}.
That means there are exactly two similarity classes of curvilinear triangles:

  T1(k): circles (C_k, C_{k+1}, C_{k+8})
  T2(k): circles (C_k, C_{k+7}, C_{k+8})

Each triangle in class k is a scaled copy of the corresponding k=0 triangle by
factor s^k, so areas scale by s^{2k}. Thus the desired total area is

  (A1 + A2) * sum_{k>=0} s^{2k} = (A1 + A2) / (1 - s^2),

where A1 is the area between circles of radii (1, s, s^8) and A2 is the area
between radii (1, s^7, s^8).

Computing a curvilinear triangle area
-------------------------------------
For three externally tangent circles with radii r1, r2, r3, the centres form a
Euclidean triangle with side lengths (r1+r2), (r1+r3), (r2+r3). The curvilinear
triangle area equals:

  area(centre triangle) - sum(sector areas),

where each sector angle equals the corresponding triangle angle at that centre.
"""

import math
from typing import Tuple


def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _h(s: float, theta: float, n: int) -> float:
    """Return |1 - (s e^{i theta})^n|^2 / (1 + s^n)^2, using only cos."""
    sn = s**n
    num = 1.0 + (sn * sn) - 2.0 * sn * math.cos(n * theta)
    den = (1.0 + sn) ** 2
    return num / den


def _f(s: float, theta: float) -> Tuple[float, float]:
    h1 = _h(s, theta, 1)
    return (h1 - _h(s, theta, 7), h1 - _h(s, theta, 8))


def _objective(s: float, theta: float) -> float:
    f1, f2 = _f(s, theta)
    return f1 * f1 + f2 * f2


def _find_initial_guess() -> Tuple[float, float]:
    # Restrict to a range that avoids spurious small-s roots and matches the
    # non-overlapping spiral configuration.
    best = (0.9, 0.8)
    best_val = float("inf")

    s_min, s_max, ds = 0.75, 0.99, 0.002
    t_min, t_max, dt = 0.05, math.pi - 0.05, 0.01

    s = s_min
    while s <= s_max + 1e-12:
        theta = t_min
        while theta <= t_max + 1e-12:
            val = _objective(s, theta)
            if val < best_val:
                best_val = val
                best = (s, theta)
            theta += dt
        s += ds

    return best


def _newton_solve(s0: float, theta0: float) -> Tuple[float, float]:
    s, theta = s0, theta0
    for _ in range(60):
        f1, f2 = _f(s, theta)
        if max(abs(f1), abs(f2)) < 1e-15:
            return s, theta

        # Finite-difference Jacobian (central differences).
        ds = 1e-8
        dt = 1e-8

        f1_sp, f2_sp = _f(s + ds, theta)
        f1_sm, f2_sm = _f(s - ds, theta)
        f1_tp, f2_tp = _f(s, theta + dt)
        f1_tm, f2_tm = _f(s, theta - dt)

        a = (f1_sp - f1_sm) / (2.0 * ds)  # df1/ds
        b = (f1_tp - f1_tm) / (2.0 * dt)  # df1/dtheta
        c = (f2_sp - f2_sm) / (2.0 * ds)  # df2/ds
        d = (f2_tp - f2_tm) / (2.0 * dt)  # df2/dtheta

        det = a * d - b * c
        if det == 0.0 or not math.isfinite(det):
            # Fallback: restart from a nearby point.
            s *= 0.999
            theta *= 0.999
            continue

        delta_s = (d * f1 - b * f2) / det
        delta_t = (-c * f1 + a * f2) / det

        # Damped update to stay within bounds and decrease residual.
        cur_obj = f1 * f1 + f2 * f2
        step = 1.0
        for _ls in range(40):
            ns = s - step * delta_s
            nt = theta - step * delta_t
            if not (0.0 < ns < 1.0 and 0.0 < nt < math.pi):
                step *= 0.5
                continue
            nobj = _objective(ns, nt)
            if nobj < cur_obj:
                s, theta = ns, nt
                break
            step *= 0.5
        else:
            # Could not improve; stop.
            break

    return s, theta


def _curvilinear_triangle_area(r1: float, r2: float, r3: float) -> float:
    """Area between three externally tangent circles of radii r1,r2,r3."""
    # Triangle of centres has side lengths r_i + r_j.
    a = r2 + r3  # opposite r1-centre
    b = r1 + r3  # opposite r2-centre
    c = r1 + r2  # opposite r3-centre

    p = 0.5 * (a + b + c)
    tri_sq = p * (p - a) * (p - b) * (p - c)
    tri_sq = max(0.0, tri_sq)
    tri_area = math.sqrt(tri_sq)

    # Angles at the centres (law of cosines), clamped for numerical safety.
    cos1 = (b * b + c * c - a * a) / (2.0 * b * c)
    cos2 = (a * a + c * c - b * b) / (2.0 * a * c)
    cos3 = (a * a + b * b - c * c) / (2.0 * a * b)
    ang1 = math.acos(_clamp(cos1, -1.0, 1.0))
    ang2 = math.acos(_clamp(cos2, -1.0, 1.0))
    ang3 = math.acos(_clamp(cos3, -1.0, 1.0))

    sectors = 0.5 * (r1 * r1 * ang1 + r2 * r2 * ang2 + r3 * r3 * ang3)
    return tri_area - sectors


def solve() -> str:
    s0, t0 = _find_initial_guess()
    s, theta = _newton_solve(s0, t0)

    # Basic sanity checks (do not embed any known answers).
    f1, f2 = _f(s, theta)
    assert 0.0 < s < 1.0
    assert 0.0 < theta < math.pi
    assert abs(f1) < 1e-12 and abs(f2) < 1e-12

    # Base curvilinear triangle areas.
    s7 = s**7
    s8 = s7 * s
    a0 = _curvilinear_triangle_area(1.0, s, s8)
    b0 = _curvilinear_triangle_area(1.0, s7, s8)

    total = (a0 + b0) / (1.0 - s * s)
    return f"{total:.10f}"


if __name__ == "__main__":
    print(solve())
