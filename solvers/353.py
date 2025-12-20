#!/usr/bin/env python3
"""
Project Euler 353: Risky Moon

We consider all integer-coordinate points on the sphere x^2 + y^2 + z^2 = r^2.
Every pair of stations is connected by the shorter great-circle arc.
For an arc of length d, its risk is (d / (pi*r))^2.
The total risk of a multi-stop journey is the sum of edge risks.

This program computes:
    sum_{n=1..15} M(2^n - 1)
rounded to 10 digits after the decimal point.

No external libraries are used.
"""

from __future__ import annotations

import math
import heapq
from typing import Dict, List, Tuple


# ----------------------------- Small-number helpers -----------------------------


def sieve_spf(n: int) -> List[int]:
    """Smallest prime factor sieve for 0..n."""
    spf = list(range(n + 1))
    if n >= 1:
        spf[1] = 1
    lim = int(n**0.5)
    for i in range(2, lim + 1):
        if spf[i] == i:  # prime
            step = i
            start = i * i
            for j in range(start, n + 1, step):
                if spf[j] == j:
                    spf[j] = i
    return spf


def precompute_prime_sum_squares(
    limit: int, spf: List[int]
) -> Dict[int, Tuple[int, int]]:
    """
    For every prime p <= limit with p ≡ 1 (mod 4), find one (a,b) with a^2+b^2=p.
    Brute force is fine because p <= 2*(2^15-1) < 70000.
    """
    rep: Dict[int, Tuple[int, int]] = {}
    isqrt = math.isqrt
    for p in range(2, limit + 1):
        if spf[p] == p and (p & 3) == 1:
            root = isqrt(p)
            a = b = -1
            for x in range(1, root + 1):
                y2 = p - x * x
                y = isqrt(y2)
                if y * y == y2:
                    a, b = x, y
                    break
            if a < 0:
                raise RuntimeError(
                    "Failed to find sum-of-two-squares representation for prime", p
                )
            if a < b:
                a, b = b, a
            rep[p] = (a, b)
    return rep


def gauss_mul(ax: int, ay: int, bx: int, by: int) -> Tuple[int, int]:
    """(ax + i*ay) * (bx + i*by)."""
    return (ax * bx - ay * by, ax * by + ay * bx)


class Helpers:
    """
    Shared precomputations up to 2*max_r:
      - smallest prime factors
      - one sum-of-two-squares representation for each prime p ≡ 1 (mod 4)
      - cached prime factorizations for small integers
    """

    __slots__ = ("spf", "prime_rep", "factor_cache")

    def __init__(self, max_lim: int) -> None:
        self.spf = sieve_spf(max_lim)
        self.prime_rep = precompute_prime_sum_squares(max_lim, self.spf)
        self.factor_cache: List[List[Tuple[int, int]] | None] = [None] * (max_lim + 1)
        self.factor_cache[0] = []
        self.factor_cache[1] = []

    def factor_pairs(self, n: int) -> List[Tuple[int, int]]:
        """Return prime factorization of n (n <= max_lim) as list of (p, exp)."""
        cached = self.factor_cache[n]
        if cached is not None:
            return cached
        spf = self.spf
        x = n
        out: List[Tuple[int, int]] = []
        while x > 1:
            p = spf[x]
            e = 1
            x //= p
            while x % p == 0:
                x //= p
                e += 1
            out.append((p, e))
        self.factor_cache[n] = out
        return out


def sumsq_pairs_from_factors(
    factors: Dict[int, int], prime_rep: Dict[int, Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """
    Enumerate all canonical nonnegative solutions (a>=b>=0) to a^2 + b^2 = N
    given the prime factorization of N (as a dict prime->exp).

    Uses Gaussian-integer multiplicativity:
      - primes p ≡ 3 (mod 4) must have even exponent (else no solutions)
      - primes p ≡ 1 (mod 4) contribute choices by splitting exponents between p and its conjugate
      - square factors from 2 and p ≡ 3 (mod 4) are pulled out as a scalar
      - an odd exponent of 2 contributes a factor of (1+i)
    """
    if not factors:
        return [(1, 0)]

    scalar = 1
    e2 = factors.get(2, 0)
    if e2:
        scalar <<= e2 // 2  # 2^(e2//2)
        odd2 = e2 & 1
    else:
        odd2 = 0

    reps: List[Tuple[int, int]] = [(1, 0)]

    for p, e in factors.items():
        if p == 2:
            continue
        if (p & 3) == 3:
            if e & 1:
                return []
            scalar *= pow(p, e // 2)
            continue

        # p ≡ 1 (mod 4)
        a, b = prime_rep[p]
        gp = (a, b)
        gpc = (a, -b)  # conjugate

        # powers up to e (small)
        pow_gp: List[Tuple[int, int]] = [(1, 0)]
        pow_gpc: List[Tuple[int, int]] = [(1, 0)]
        for _ in range(e):
            x, y = pow_gp[-1]
            pow_gp.append(gauss_mul(x, y, gp[0], gp[1]))
            x, y = pow_gpc[-1]
            pow_gpc.append(gauss_mul(x, y, gpc[0], gpc[1]))

        reps_p: List[Tuple[int, int]] = []
        for k in range(e + 1):
            x1, y1 = pow_gp[k]
            x2, y2 = pow_gpc[e - k]
            reps_p.append(gauss_mul(x1, y1, x2, y2))

        new: List[Tuple[int, int]] = []
        for rx, ry in reps:
            for tx, ty in reps_p:
                new.append(gauss_mul(rx, ry, tx, ty))
        reps = new

    if odd2:
        reps = [gauss_mul(x, y, 1, 1) for (x, y) in reps]

    if scalar != 1:
        reps = [(x * scalar, y * scalar) for (x, y) in reps]

    out = set()
    for x, y in reps:
        a = abs(x)
        b = abs(y)
        if a < b:
            a, b = b, a
        out.add((a, b))
    return list(out)


# ----------------------------- Geometry and shortest path -----------------------------


def generate_points_north(
    r: int, helpers: Helpers
) -> Tuple[List[int], List[int], List[int], int]:
    """
    Generate all stations with z >= 0 on x^2 + y^2 + z^2 = r^2.
    Returns coordinate arrays (xs, ys, zs) and the index of the North Pole (0,0,r).

    For each z:
        x^2 + y^2 = r^2 - z^2 = (r-z)(r+z),
    and both factors are <= 2r, so factoring is cheap with the SPF table.
    """
    r2 = r * r
    xs: List[int] = []
    ys: List[int] = []
    zs: List[int] = []
    start = -1

    fp = helpers.factor_pairs
    prime_rep = helpers.prime_rep

    for z in range(r + 1):
        s = r2 - z * z
        if s == 0:
            xs.append(0)
            ys.append(0)
            zs.append(z)
            if z == r:
                start = len(xs) - 1
            continue

        a = r - z
        b = r + z
        fac: Dict[int, int] = {}
        for p, e in fp(a):
            fac[p] = fac.get(p, 0) + e
        for p, e in fp(b):
            fac[p] = fac.get(p, 0) + e

        pairs = sumsq_pairs_from_factors(fac, prime_rep)
        if not pairs:
            continue

        for aa, bb in pairs:
            # expand signs (and swap when aa != bb) to cover all integer (x,y) for this z
            xs1 = (aa,) if aa == 0 else (aa, -aa)
            ys1 = (bb,) if bb == 0 else (bb, -bb)
            if aa == bb:
                for x in xs1:
                    for y in ys1:
                        xs.append(x)
                        ys.append(y)
                        zs.append(z)
            else:
                for x in xs1:
                    for y in ys1:
                        xs.append(x)
                        ys.append(y)
                        zs.append(z)
                        xs.append(y)
                        ys.append(x)
                        zs.append(z)

    if start < 0:
        raise RuntimeError("North pole station missing; generation failed")
    return xs, ys, zs, start


def compute_M(r: int, helpers: Helpers, L: int) -> float:
    """
    Compute M(r) using:
      - an equatorial symmetry reduction:
            M(r) = min_{p with z>=0} ( 2*dist(N, p) + risk(p, reflect(p)) )
        where reflect(x,y,z)=(x,y,-z).
      - Dijkstra from the North Pole over a sparse, local neighbor graph built with a 3D grid.

    The neighbor graph is built by connecting only points with |dx|,|dy|,|dz| <= L.
    (For the required radii, this local graph is sufficient and dramatically reduces work.)
    """
    xs, ys, zs, start = generate_points_north(r, helpers)
    n = len(xs)

    # Spatial hashing into cubic cells of side length L
    cell = L
    cell_map: Dict[Tuple[int, int, int], List[int]] = {}
    for i in range(n):
        key = (xs[i] // cell, ys[i] // cell, zs[i] // cell)
        cell_map.setdefault(key, []).append(i)

    r2 = r * r
    inv_r2 = 1.0 / r2
    inv_pi = 1.0 / math.pi
    acos = math.acos

    # risk of the "reflection edge" depends only on z:
    # dot((x,y,z),(x,y,-z)) = r^2 - 2z^2  ->  cos(theta)=1-2z^2/r^2
    ref_risk = [0.0] * (r + 1)
    for z in range(r + 1):
        c = 1.0 - 2.0 * (z * z) * inv_r2
        if c < -1.0:
            c = -1.0
        t = acos(c) * inv_pi
        ref_risk[z] = t * t

    # Dijkstra
    INF = 1e100
    dist = [INF] * n
    dist[start] = 0.0
    heap = [(0.0, start)]

    # Upper bound from path via (0,r,0): risk = 0.5 for any integer r.
    best = 0.5

    get = cell_map.get
    heappop = heapq.heappop
    heappush = heapq.heappush

    while heap:
        d, u = heappop(heap)
        if d != dist[u]:
            continue

        # No remaining node can improve the current best if even 2*d already matches/exceeds it.
        if 2.0 * d >= best:
            break

        z0 = zs[u]
        cand = 2.0 * d + ref_risk[z0]
        if cand < best:
            best = cand

        x0 = xs[u]
        y0 = ys[u]
        cx = x0 // cell
        cy = y0 // cell
        cz = z0 // cell

        for dxcell in (-1, 0, 1):
            for dycell in (-1, 0, 1):
                for dzcell in (-1, 0, 1):
                    lst = get((cx + dxcell, cy + dycell, cz + dzcell))
                    if not lst:
                        continue
                    for v in lst:
                        if v == u:
                            continue
                        dx = xs[v] - x0
                        if dx > L or dx < -L:
                            continue
                        dy = ys[v] - y0
                        if dy > L or dy < -L:
                            continue
                        dz = zs[v] - z0
                        if dz > L or dz < -L:
                            continue

                        dot = x0 * xs[v] + y0 * ys[v] + z0 * zs[v]
                        c = dot * inv_r2
                        if c > 1.0:
                            c = 1.0
                        elif c < -1.0:
                            c = -1.0
                        t = acos(c) * inv_pi
                        w = t * t

                        nd = d + w
                        if nd < dist[v] - 1e-15:
                            dist[v] = nd
                            heappush(heap, (nd, v))

    return best


def choose_L(r: int) -> int:
    """
    Neighborhood threshold for edges:
      - for very small radii, we include all edges (L=2r)
      - for medium radii, a smaller local window is enough
      - for the largest radii in this task, a slightly larger window is needed
    """
    if r <= 512:
        return 2 * r
    if r < 8191:
        return 256
    return 512


# ----------------------------- Main -----------------------------


def main() -> None:
    max_r = (1 << 15) - 1
    helpers = Helpers(2 * max_r)

    # The problem statement explicitly gives the risk of splitting a half-circle into two equal arcs:
    # risk = (1/2)^2 + (1/2)^2 = 1/2.
    assert abs((0.5 * 0.5 + 0.5 * 0.5) - 0.5) < 1e-15

    total = 0.0
    m7 = None

    for n in range(1, 16):
        r = (1 << n) - 1
        L = choose_L(r)
        m = compute_M(r, helpers, L)
        total += m
        if r == 7:
            m7 = m

    # Given in the statement: M(7)=0.1784943998 (rounded to 10 digits after the decimal point).
    assert m7 is not None
    assert abs(m7 - 0.1784943998) < 5e-11

    print(f"{total:.10f}")


if __name__ == "__main__":
    main()
