#!/usr/bin/env python3
"""
Project Euler 904: Pythagorean Angle

Compute F(N, L) where for each alpha we select the integer right triangle (hypotenuse <= L)
whose "median angle" theta is closest to alpha (degrees). Ties (same minimum difference)
are resolved by choosing the triangle with the largest area.

No external libraries are used. Single core, no multithreading.
"""

import math
from math import gcd


_DEG2RAD = math.pi / 180.0
# g(t) is maximized at t = sqrt(2) - 1.
_T0 = math.sqrt(2.0) - 1.0


def _g_of_t(t: float) -> float:
    """g(t) = tan(theta) when t = n/m for Euclid parameters (m>n>0)."""
    tt = t * t
    denom = 1.0 + tt
    return 3.0 * t * (1.0 - tt) / (denom * denom)


def _root_left(y: float) -> float:
    """Solve g(t)=y on [0, t0] (increasing), by binary search."""
    lo, hi = 0.0, _T0
    for _ in range(80):
        mid = (lo + hi) * 0.5
        if _g_of_t(mid) < y:
            lo = mid
        else:
            hi = mid
    return (lo + hi) * 0.5


def _root_right(y: float) -> float:
    """Solve g(t)=y on [t0, 1] (decreasing), by binary search."""
    lo, hi = _T0, 1.0
    for _ in range(80):
        mid = (lo + hi) * 0.5
        if _g_of_t(mid) > y:
            lo = mid
        else:
            hi = mid
    return (lo + hi) * 0.5


def _cf_candidates_circle(x: float, L: int):
    """
    Continued-fraction candidates for a real x in (0,1), under the constraint p^2 + q^2 <= L.

    We generate convergents until the next convergent would violate the circle constraint.
    At that final step, the best admissible semiconvergents have the form:

        (p, q) = (k*p1 + p0,  k*q1 + q0)   for 1 <= k <= a-1,

    where p1/q1 and p0/q0 are the last two convergents and 'a' is the next CF coefficient.
    We choose k near the largest one satisfying (k*p1+p0)^2 + (k*q1+q0)^2 <= L.
    """
    cands = set()

    p0, q0 = 0, 1
    p1, q1 = 1, 0
    frac = x

    for _ in range(80):
        a = int(frac)
        p2 = a * p1 + p0
        q2 = a * q1 + q0

        if p2 * p2 + q2 * q2 > L:
            # Find the largest k (<= a-1) such that:
            # (p0+k*p1)^2 + (q0+k*q1)^2 <= L.
            A = p1 * p1 + q1 * q1
            B = 2 * (p0 * p1 + q0 * q1)
            C = p0 * p0 + q0 * q0 - L

            kmax = 0
            if A > 0:
                disc = B * B - 4 * A * C
                if disc > 0:
                    s = math.isqrt(disc)
                    # positive root of A*k^2 + B*k + C = 0
                    kmax = (-B + s) // (2 * A)

            kmax = min(kmax, a - 1)

            # Check a tiny neighborhood around kmax.
            lo = max(1, kmax - 3)
            hi = min(a - 1, kmax + 3)
            for k in range(lo, hi + 1):
                ps = k * p1 + p0
                qs = k * q1 + q0
                if ps > 0 and qs > 0 and ps * ps + qs * qs <= L:
                    cands.add((ps, qs))
            break

        if p2 > 0 and q2 > 0:
            cands.add((p2, q2))

        if frac == a:
            break
        frac = 1.0 / (frac - a)
        p0, q0, p1, q1 = p1, q1, p2, q2

    # Also include the last convergent if admissible.
    if p1 > 0 and q1 > 0 and p1 * p1 + q1 * q1 <= L:
        cands.add((p1, q1))

    return cands


def _triangle_from_mn(m: int, n: int):
    """
    Construct Euclid triple from (m,n), then reduce by gcd(a,b,c) so we work with the
    underlying primitive shape even if (m,n) is not a primitive generator.
    Returns (a,b,c) with a,b legs and c hypotenuse.
    """
    a = m * m - n * n
    b = 2 * m * n
    c = m * m + n * n
    g = gcd(a, gcd(b, c))
    return a // g, b // g, c // g


def _tan_theta_from_legs(a: int, b: int) -> float:
    """tan(theta) = 3ab / (2(a^2 + b^2)) for the median angle in a right triangle."""
    aa = a * a
    bb = b * b
    return (3.0 * a * b) / (2.0 * (aa + bb))


def f(alpha_deg: float, L: int) -> int:
    """
    Return f(alpha, L): the perimeter of the chosen triangle for angle alpha (degrees)
    and hypotenuse bound L, following the minimization + tie-break rules.
    """
    alpha_rad = alpha_deg * _DEG2RAD
    y = math.tan(alpha_rad)

    # Two real solutions for t in (0,1) to g(t)=y (one on each side of the maximum).
    r1 = _root_left(y)
    r2 = _root_right(y)

    best_diff = float("inf")
    best_area_key = -1  # compare k^2 * a * b
    best_perim = 0

    for r in (r1, r2):
        for n, m in _cf_candidates_circle(r, L):
            if not (0 < n < m):
                continue

            a, b, c = _triangle_from_mn(m, n)
            if a <= 0 or b <= 0:
                continue
            if c > L:
                continue

            # Always scale to the largest allowed triangle for this primitive shape
            # (theta is scale-invariant; area tie-break prefers larger scaling).
            k = L // c
            if k <= 0:
                continue

            theta = math.atan(_tan_theta_from_legs(a, b))
            diff = abs(theta - alpha_rad)

            area_key = (k * k) * a * b
            perim = k * (a + b + c)

            if diff + 1e-16 < best_diff:
                best_diff = diff
                best_area_key = area_key
                best_perim = perim
            elif abs(diff - best_diff) <= 1e-16:
                if area_key > best_area_key:
                    best_area_key = area_key
                    best_perim = perim

    return best_perim


def F(N: int, L: int) -> int:
    """Compute F(N, L) = sum_{n=1..N} f(cuberoot(n), L)."""
    total = 0
    one_third = 1.0 / 3.0
    for n in range(1, N + 1):
        alpha = n**one_third
        total += f(alpha, L)
    return total


def _self_test():
    # Examples from the problem statement:
    assert f(30.0, 10**2) == 198
    assert f(10.0, 10**6) == 1600158
    assert F(10, 10**6) == 16684370


def main():
    _self_test()
    print(F(45000, 10**10))


if __name__ == "__main__":
    main()
