#!/usr/bin/env python3
"""Project Euler 747

Count all valid ways to cut a triangular pizza into equal-area triangular pieces.

We consider all integers n >= 3. A configuration is formed by choosing an interior
point P and drawing n straight segments from P to the boundary so that the n
resulting pieces are triangles of equal area.

Let ψ(n) be the number of configurations for a fixed n, and let
Ψ(m) = sum_{n=3..m} ψ(n).

This program prints Ψ(10^8) modulo 1_000_000_007.

No external libraries are used (only Python's standard library).
"""

from __future__ import annotations

import math


MOD = 1_000_000_007


def _easy_prefix(m: int, mod: int | None) -> int:
    """
    Sum over n=3..m of:
        C(n-1, 2) + 6*(n-2)
    This covers all configurations where at least two corners are directly connected
    to the interior point (the "easy" skeletons).
    Closed form:
        (m^3 + 15 m^2 - 52 m + 36) / 6   for m >= 3, else 0
    """
    if m < 3:
        return 0
    # Use integer arithmetic; division by 6 is exact for all integers m.
    m2 = m * m
    num = m2 * m + 15 * m2 - 52 * m + 36
    val = num // 6
    return val if mod is None else val % mod


def _min_n_and_square(x: int, y: int) -> tuple[int, int]:
    """
    Hard case parameterization (x,y) for one fixed uncut vertex:

    Let D = x*y*(x+1)*(y+1).
    Then the smallest n for which a valid configuration exists is:
        n_min = 2xy + x + y + 1 + ceil( 2*sqrt(D) )
    If D is a perfect square then at n = n_min there is exactly one solution,
    otherwise there are two.

    Returns (n_min, is_square_D).
    """
    # Compute ceil(2*sqrt(D)) exactly via integer sqrt on 4D.
    # 4D = 4*x*(x+1)*y*(y+1)
    four_d = 4 * x * (x + 1) * y * (y + 1)
    r = math.isqrt(four_d)
    if r * r == four_d:
        ceil2 = r
        sq = 1
    else:
        ceil2 = r + 1
        sq = 0
    n_min = 2 * x * y + x + y + 1 + ceil2
    return n_min, sq


def _y_max_for_x(m: int, x: int) -> int:
    """
    For fixed x>=1 in the hard case, find the maximum y>=x such that n_min(x,y) <= m.
    If no y works, return x-1.
    """
    # A safe upper bound:
    # n_min > 4*x*y, hence y <= (m-1)//(4*x).
    # We use a slightly looser bound that avoids division by zero for small m.
    if 4 * x > m - 1:
        return x - 1
    hi = (m - 1) // (4 * x) + 2
    if hi < x:
        hi = x
    # Binary search on y in [x, hi] for max satisfying n_min <= m
    lo = x
    ok = x - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        n_min, _ = _min_n_and_square(x, mid)
        if n_min <= m:
            ok = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return ok


def _hard_prefix(m: int, mod: int | None) -> int:
    """
    Hard skeleton count for one fixed uncut vertex:
        A(m) = sum over x>=1,y>=1 of contributions for all n<=m.

    For each (x,y), valid configurations exist for every n >= n_min(x,y).
    If D is not a square, there are 2 configurations at each such n.
    If D is a square, there is 1 configuration at n=n_min and 2 afterwards.

    Contribution up to m:
        if n_min > m: 0
        else: 2*(m - n_min + 1) - is_square(D)

    The total contribution to Ψ is 3*A(m) (three choices of the uncut vertex).
    We exploit symmetry in (x,y) to halve work:
        sum_{x<y} counted twice, diagonal once.
    """
    if m < 3:
        return 0

    # If n_min(x,y) <= m, then 4xy < m, so min(x,y) <= floor(sqrt((m-1)//4)).
    # In the symmetric sum with x<=y, it suffices to iterate x up to that bound.
    k = (m - 1) // 4
    if k <= 0:
        return 0
    x_max = math.isqrt(k)

    total = 0
    isqrt = math.isqrt
    if mod is None:
        cutoff = 0
    else:
        # Reduce occasionally to avoid doing '%' on every inner-loop iteration.
        cutoff = mod << 20
    for x in range(1, x_max + 1):
        y_max = _y_max_for_x(m, x)
        if y_max < x:
            continue

        # Precompute x-dependent pieces
        A = x * (x + 1)

        # Start y at x and increment; update y(y+1) and 2xy incrementally.
        y = x
        yy1 = y * (y + 1)
        two_xy = 2 * x * y

        while y <= y_max:
            # 4D = 4*A*yy1
            four_d = (A * yy1) << 2
            r = isqrt(four_d)
            sq = 1 if r * r == four_d else 0
            ceil2 = r if sq else r + 1

            n_min = two_xy + x + y + 1 + ceil2
            # n_min must be <= m by construction of y_max; keep the check for safety.
            if n_min <= m:
                cnt = 2 * (m - n_min + 1) - sq
                if x == y:
                    add = cnt
                else:
                    add = cnt << 1  # multiply by 2 for (x,y) and (y,x)
                if mod is None:
                    total += add
                else:
                    total += add
                    if total >= cutoff:
                        total %= mod

            # advance y -> y+1
            # yy1(y) = y(y+1); yy1(y+1) = (y+1)(y+2) = yy1 + 2y + 2
            yy1 += (y << 1) + 2
            y += 1
            two_xy += x << 1

    return total if mod is None else total % mod


def Psi(m: int, mod: int | None = MOD) -> int:
    """
    Computes Ψ(m) = sum_{n=3..m} ψ(n) for the problem.
    If mod is None, returns the exact integer (only practical for small m).
    Otherwise returns modulo mod.
    """
    easy = _easy_prefix(m, mod)
    hard_one_vertex = _hard_prefix(m, mod)
    if mod is None:
        return easy + 3 * hard_one_vertex
    return (easy + 3 * hard_one_vertex) % mod


def psi(n: int) -> int:
    """Returns ψ(n) for small n (exact integer)."""
    if n < 3:
        return 0
    return Psi(n, None) - Psi(n - 1, None)


def _run_self_tests() -> None:
    # Test values from the problem statement (add asserts as requested).
    assert psi(3) == 7
    assert psi(6) == 34
    assert psi(10) == 90
    assert Psi(10, None) == 345
    assert Psi(1000, None) == 172166601


def main() -> None:
    _run_self_tests()
    m = 100_000_000
    print(Psi(m, MOD))


if __name__ == "__main__":
    main()
