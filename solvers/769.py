#!/usr/bin/env python3
"""
Project Euler 769

Count primitive representations (x, y) of squares by
    f(x, y) = x^2 + 5xy + 3y^2
i.e. f(x, y) = z^2 with z <= N, x,y>0, gcd(x,y)=1.

No external libraries are used (only Python stdlib).
"""

from __future__ import annotations

import math
from array import array


SQRT3 = math.sqrt(3.0)

# Precompute inverses mod 13 for residues 1..12 (13 is prime)
INV_MOD13 = [0] * 13
for a in range(1, 13):
    INV_MOD13[a] = pow(a, -1, 13)


def build_spf_linear(n: int) -> array:
    """
    Build smallest prime factor table up to n (inclusive) using a linear sieve.
    Memory: ~4*(n+1) bytes.
    """
    spf = array("I", [0]) * (n + 1)
    primes: list[int] = []
    for i in range(2, n + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
        si = spf[i]
        for p in primes:
            v = i * p
            if v > n:
                break
            spf[v] = p
            if p == si:
                break
    return spf


def distinct_prime_factors(n: int, spf: array) -> list[int]:
    """Return the distinct prime factors of n."""
    res: list[int] = []
    while n > 1:
        p = spf[n]
        res.append(p)
        while n % p == 0:
            n //= p
    return res


def count_coprime_interval_with_divs(
    n: int, L: int, R: int, spf: array
) -> tuple[int, list[int], list[int]]:
    """
    Count integers q in [L, R] such that gcd(n, q) = 1, using inclusion-exclusion
    over the distinct prime factors of n.

    Also returns (ds, mus) where ds are squarefree divisors of n and mus are their
    Möbius values (±1) in the same order, so that:
        count = sum_{i} mus[i] * (floor(R/ds[i]) - floor((L-1)/ds[i]))
    """
    primes = distinct_prime_factors(n, spf)
    ds = [1]
    mus = [1]
    for pr in primes:
        m = len(ds)
        for i in range(m):
            ds.append(ds[i] * pr)
            mus.append(-mus[i])

    Lm = L - 1
    total = 0
    for d, mu in zip(ds, mus):
        total += mu * (R // d - Lm // d)
    return total, ds, mus


def _count_congruence_in_interval(L: int, R: int, mod: int, rem: int) -> int:
    """
    Count integers x in [L, R] with x ≡ rem (mod mod), assuming 0 <= rem < mod.
    """
    if rem < L:
        rem += ((L - rem + mod - 1) // mod) * mod
    if rem > R:
        return 0
    return 1 + (R - rem) // mod


def count_coprime_with_mod13_congruence(
    ds: list[int],
    mus: list[int],
    L: int,
    R: int,
    rem13: int,
) -> int:
    """
    Count integers q in [L, R] such that:
      - q ≡ rem13 (mod 13)
      - and gcd(q, n) = 1, where (ds, mus) are squarefree divisors / Möbius values
        for that n.

    Uses inclusion-exclusion with the additional congruence.
    """
    total = 0
    for d, mu in zip(ds, mus):
        # In our usage, gcd(d,13)=1, so d%13 is 1..12.
        inv = INV_MOD13[d % 13]
        m0 = (rem13 * inv) % 13
        rem = d * m0  # unique solution modulo 13*d
        mod = 13 * d
        total += mu * _count_congruence_in_interval(L, R, mod, rem)
    return total


def max_abs_p_for_negative_branch(N: int) -> int:
    """
    For p = -a < 0, q is restricted by the sign conditions:
        sqrt(3)*a < q < (5/2)*a
    and z = -(q^2 - 5aq + 3a^2) must satisfy z <= N.

    This function finds the largest a for which there exists an integer q
    satisfying all constraints. Monotonicity lets us binary search.
    """
    hi = int(math.isqrt(N)) + 2
    lo = 0
    while lo + 1 < hi:
        a = (lo + hi) // 2
        if a == 0:
            lo = a
            continue

        qmin = int(SQRT3 * a) + 1
        thr = 3 * a * a
        while qmin * qmin <= thr:
            qmin += 1

        qmax = (5 * a - 1) // 2  # strict q < 2.5a
        if qmin > qmax:
            ok = False
        else:
            z = -(qmin * qmin - 5 * a * qmin + 3 * a * a)
            ok = z <= N

        if ok:
            lo = a
        else:
            hi = a
    return lo


def C(N: int, spf: array) -> int:
    """
    Compute C(N) as defined in the problem statement.
    """
    N = int(N)
    fourN = 4 * N
    total = 0

    # Branch 1: p > 0, q > sqrt(3)*p, z = q^2 + 5pq + 3p^2
    pmax = int(math.isqrt(N // 3))
    isqrt = math.isqrt

    for p in range(1, pmax + 1):
        qmin = int(SQRT3 * p) + 1
        thr = 3 * p * p
        while qmin * qmin <= thr:
            qmin += 1

        # Solve q^2 + 5pq + 3p^2 <= N for q (upper root).
        disc = 13 * p * p + fourN
        qmax = (isqrt(disc) - 5 * p) // 2
        if qmax < qmin:
            continue

        # Safety adjustment (rare): ensure z(qmax) <= N
        pp3 = 3 * p * p
        while qmax >= qmin and (qmax * qmax + 5 * p * qmax + pp3) > N:
            qmax -= 1
        if qmax < qmin:
            continue

        cnt, ds, mus = count_coprime_interval_with_divs(p, qmin, qmax, spf)

        # Non-primitive cases are exactly those with q ≡ 4p (mod 13) (and p not multiple of 13).
        if p % 13 != 0:
            bad = count_coprime_with_mod13_congruence(ds, mus, qmin, qmax, (4 * p) % 13)
            cnt -= bad

        total += cnt

    # Branch 2: p = -a < 0, sqrt(3)*a < q < 2.5a, z = -(q^2 - 5aq + 3a^2)
    amax = max_abs_p_for_negative_branch(N)
    threshold = int(
        isqrt(fourN // 13)
    )  # where discriminant 13a^2 - 4N becomes nonnegative

    for a in range(1, amax + 1):
        qmin = int(SQRT3 * a) + 1
        thr = 3 * a * a
        while qmin * qmin <= thr:
            qmin += 1

        qmax = (5 * a - 1) // 2  # strict q < 2.5a
        if qmax < qmin:
            continue

        # When 13a^2 > 4N, the z <= N constraint cuts off the top of the interval.
        if a > threshold:
            disc = 13 * a * a - fourN
            s = isqrt(disc)
            lim = (5 * a - s) // 2
            if lim < qmax:
                qmax = lim

        # Safety adjustment (needed because floor(isqrt) can overestimate the root here).
        while qmax >= qmin and (-(qmax * qmax - 5 * a * qmax + 3 * a * a)) > N:
            qmax -= 1
        if qmax < qmin:
            continue

        cnt, ds, mus = count_coprime_interval_with_divs(a, qmin, qmax, spf)

        if a % 13 != 0:
            # For p=-a, q ≡ 4p (mod 13) is q ≡ -4a (mod 13).
            bad = count_coprime_with_mod13_congruence(
                ds, mus, qmin, qmax, (-4 * a) % 13
            )
            cnt -= bad

        total += cnt

    return total


def solve() -> int:
    N = 10**14
    # Build SPF once for the maximum p needed at this N.
    maxp = max(int(math.isqrt(N // 3)), max_abs_p_for_negative_branch(N))
    spf = build_spf_linear(maxp)

    # Tests from the statement
    assert C(10**3, spf) == 142
    assert C(10**6, spf) == 142463

    return C(N, spf)


if __name__ == "__main__":
    print(solve())
