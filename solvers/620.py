#!/usr/bin/env python3
"""
Project Euler 620 - Planetary Gears

We count, for each integer triple (s,p,q) with s+p+q <= n, p<q, s>=5, p>=5,
how many distinct placements of four "planet" gears are possible such that all
gears mesh perfectly at 1cm pitch, and the inner/outer gears stay at least 1cm
apart at their closest point.

Key observation used here:
For fixed (s,p,q) with c = s+p+q, the "phase compatibility" for perfect meshing
reduces to a single integer condition on a continuous parameter. As the offset
between the two main gear centers varies across its allowed range, that phase
quantity varies monotonically and sweeps an interval. Therefore the number of
valid discrete arrangements equals the number of integers inside that interval,
which can be computed from its endpoints without any search.

This file avoids external libraries and uses only Python's standard `math`.
"""

from __future__ import annotations

import math
import sys


PI = math.pi
TWO_PI = 2.0 * PI
# Small positive margin to protect floor() against tiny negative floating errors.
EPS = 1e-12


def _clamp(x: float) -> float:
    """Clamp to [-1, 1] to keep acos() safe under rounding."""
    if x > 1.0:
        return 1.0
    if x < -1.0:
        return -1.0
    return x


def arrangements_for_spq(s: int, p: int, q: int) -> int:
    """
    Return g(s+p+q, s, p, q) for integers s,p,q with p<q and all >= 5.

    Derived formula:
      g = floor(D_max - D_min)
    where D_max is reached at the degenerate offset (not included) and D_min is
    reached when the inner/outer boundaries are exactly 1cm apart (included).

    After simplifying the arc-length expression, the count becomes:

      g = floor( (alpha*(p+s) + beta*(p+q+2s)) / pi )

    where alpha and beta are angles of a triangle formed by centers of:
      outer gear C, inner gear S, and a p-planet center,
    computed at the maximal allowed center offset (gap = 1cm).
    """
    # Work in "circumference-scaled" lengths (multiply geometric lengths by 2π).
    # For the triangle (C, S, P):
    #   |CP| * 2π = s + q
    #   |SP| * 2π = s + p
    #   |CS| * 2π = (p + q) - 2π      (because the radial gap is 1cm)
    a = s + q  # |CP| * 2π
    b = s + p  # |SP| * 2π
    c_len = (p + q) - TWO_PI  # |CS| * 2π (float)

    a2 = a * a
    b2 = b * b
    c2 = c_len * c_len

    # Angle at the planet center.
    cos_alpha = _clamp((a2 + b2 - c2) / (2.0 * a * b))
    alpha = math.acos(cos_alpha)

    # Angle at the outer gear center.
    cos_beta = _clamp((a2 + c2 - b2) / (2.0 * a * c_len))
    beta = math.acos(cos_beta)

    t = (alpha * b + beta * ((p + q) + 2 * s)) / PI

    # t should be non-negative, and we only need floor(t). int() truncates toward 0.
    return int(t + EPS)


def compute_G(n: int) -> int:
    """Compute G(n) as defined in the problem statement."""
    total = 0

    # Need s>=5, p>=5, q>=5, p<q, and s+p+q <= n.
    # The smallest (p,q) with p<q and both >=5 is (5,6), hence s <= n-11.
    for s in range(5, n - 9):
        rem = n - s

        # Ensure q exists: q >= p+1 and p+q <= rem  => 2p <= rem-1.
        p_max = (rem - 1) // 2
        for p in range(5, p_max + 1):
            q_max = rem - p
            for q in range(p + 1, q_max + 1):
                total += arrangements_for_spq(s, p, q)

    return total


def main(argv: list[str]) -> None:
    n = 500
    if len(argv) >= 2:
        n = int(argv[1])

    # Tests from the problem statement.
    assert arrangements_for_spq(5, 5, 6) == 9
    assert compute_G(16) == 9
    assert compute_G(20) == 205

    print(compute_G(n))


if __name__ == "__main__":
    main(sys.argv)
