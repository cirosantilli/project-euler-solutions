#!/usr/bin/env python3
"""
Project Euler 883

Counts "remarkable" triangles on the triangular/hexagonal lattice whose incenter
is a lattice point and whose inradius is at most R.

Implementation notes:
- No external libraries (standard library only)
- Single-threaded
"""
from __future__ import annotations

from array import array
from math import gcd, isqrt


def _count_hex_points_leq(B: int, cache: dict[int, int]) -> int:
    """
    Number of integer pairs (x,y) with x^2 + x*y + y^2 <= B.
    Includes the origin.

    Uses the identity:
      N(B) = 1 + 6 * sum_{d=1..B} chi(d) * floor(B/d),
    where chi is the nontrivial Dirichlet character mod 3:
      chi(d)= 1 if d≡1 (mod3), -1 if d≡2 (mod3), 0 if d≡0 (mod3).

    The sum is computed in O(sqrt(B)) blocks by grouping equal floor(B/d).
    """
    if B <= 0:
        return 1  # only the origin for B==0, and caller subtracts if needed
    hit = cache.get(B)
    if hit is not None:
        return hit

    # prefix sum of chi on 1..m:
    # count of numbers ≡1 mod3 minus count ≡2 mod3
    def chi_prefix(m: int) -> int:
        return (m + 2) // 3 - (m + 1) // 3

    total = 0
    n = B
    i = 1
    while i <= n:
        q = n // i
        j = n // q
        total += q * (chi_prefix(j) - chi_prefix(i - 1))
        i = j + 1

    res = 1 + 6 * total
    cache[B] = res
    return res


def remarkable_triangles(R_num: int, R_den: int = 1) -> int:
    """
    Compute T(R) for R = R_num / R_den.

    R may be non-integer for the small test cases.
    """
    # d := k + l - s, where s = sqrt(k^2 + l^2 - k*l) is integer for valid shapes.
    # Any nonzero lattice direction vector p has norm^2 Q(p) = x^2 + x*y + y^2.
    # Inradius formula implies:
    #   Q(p) <= floor(12 * R^2 / d^2).
    #
    # If d is not divisible by 3, only points in an index-3 sublattice are allowed;
    # that reduces to counting all lattice points with bound floor(B/3).
    #
    # We also need d such that B>=1 (since Q(p)>=1 for nonzero p):
    #   d <= floor(sqrt(12) * R) = floor(sqrt(12 * R^2)).
    Dmax = isqrt((12 * R_num * R_num) // (R_den * R_den))

    # Multiplicity of primitive (non-equilateral) shapes for each d (with k>l);
    # later we multiply by 2 to include the swapped (l,k) configuration.
    M = array("I", [0]) * (Dmax + 1)

    # -------- Family 1 (always d = 3*u*v) ----------
    # For fixed t = u*v, gcd(u,v)=1, u>v, and u mod3 != v mod3.
    # The number of such pairs depends only on the squarefree part:
    # it equals 2^{omega(t)-1}, where omega(t)=#distinct prime factors,
    # and it is nonzero only when t % 3 != 1.
    Nmax = Dmax // 3
    omega = array("B", [0]) * (Nmax + 1)
    for p in range(2, Nmax + 1):
        if omega[p] == 0:  # prime
            for k in range(p, Nmax + 1, p):
                omega[k] += 1

    for t in range(2, Nmax + 1):
        if t % 3 == 1:
            continue
        # omega[t] >= 1 for t>=2
        M[3 * t] += 1 << (omega[t] - 1)

    # -------- Family 2 (d = (u-v)(u+2v) = a(a+3v)) ----------
    # Enumerate v and a=u-v. For fixed v, a is small when v is large.
    D = Dmax
    vmax = D // 3
    for v in range(1, vmax + 1):
        # a^2 + 3v a - D <= 0  => a <= floor((sqrt(9v^2+4D) - 3v)/2)
        disc = 9 * v * v + 4 * D
        amax = (isqrt(disc) - 3 * v) // 2
        if amax <= 0:
            continue
        for a in range(1, amax + 1):
            u = v + a
            if u % 3 == v % 3:
                continue
            if gcd(u, v) != 1:
                continue
            d = a * (a + 3 * v)
            M[d] += 1

    # -------- Sum contributions over d ----------
    cache: dict[int, int] = {}
    C_num = 12 * R_num * R_num
    C_den = R_den * R_den

    total_scalene = 0
    for d in range(1, Dmax + 1):
        mult = M[d]
        if mult == 0:
            continue

        B = C_num // (C_den * d * d)
        if B == 0:
            continue

        if d % 3 == 0:
            pts = _count_hex_points_leq(B, cache) - 1
        else:
            pts = _count_hex_points_leq(B // 3, cache) - 1

        # Each primitive shape is generated with k>l; swapping k and l yields a
        # distinct triangle (a reflection across the angle bisector), so factor 2.
        total_scalene += 2 * mult * pts

    # -------- Equilateral triangles ----------
    # Fix the incenter at the origin. An equilateral triangle is determined by a
    # nonzero vertex position A with ||A|| = 2r. The three choices A, ωA, ω^2 A
    # describe the same triangle, so divide by 3.
    Beq = (4 * R_num * R_num) // (R_den * R_den)
    equi_points = _count_hex_points_leq(Beq, cache) - 1
    total_equilateral = equi_points // 3

    return total_scalene + total_equilateral


def _self_test() -> None:
    # Test values from the problem statement
    assert remarkable_triangles(1, 2) == 2
    assert remarkable_triangles(2, 1) == 44
    assert remarkable_triangles(10, 1) == 1302


def main() -> None:
    _self_test()
    # Problem asks for R = 10^6
    print(remarkable_triangles(10**6, 1))


if __name__ == "__main__":
    main()
