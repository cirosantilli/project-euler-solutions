#!/usr/bin/env python3
"""
Project Euler 994 - Counting Triangles

The program prints T(1234*10^8, 2345*10^8) modulo 1_000_000_007.

No final numeric answer is embedded here; the program computes it.
"""

from array import array
import os

MOD = 1_000_000_007
INV2 = (MOD + 1) // 2
INV6 = pow(6, MOD - 2, MOD)


def c2_mod(x: int) -> int:
    x %= MOD
    return x * ((x - 1) % MOD) % MOD * INV2 % MOD


def c3_mod(x: int) -> int:
    x %= MOD
    return x * ((x - 1) % MOD) % MOD * ((x - 2) % MOD) % MOD * INV6 % MOD


def p1(n: int) -> int:
    """1 + ... + n, modulo MOD."""
    n %= MOD
    return n * ((n + 1) % MOD) % MOD * INV2 % MOD


def p2(n: int) -> int:
    """1^2 + ... + n^2, modulo MOD."""
    n %= MOD
    return n * ((n + 1) % MOD) % MOD * ((2 * n + 1) % MOD) % MOD * INV6 % MOD


def p3(n: int) -> int:
    """1^3 + ... + n^3, modulo MOD."""
    s = p1(n)
    return s * s % MOD


class TotientPrefix:
    """
    Summatory values of phi(d), d*phi(d), and d^2*phi(d), modulo MOD.

    Values up to `limit` are sieved directly. Larger values are evaluated with a
    Du Jiao-style summatory recursion. For k = 0,1,2, define

        F_k(n) = sum_{d <= n} d^k phi(d).

    Since (d^k phi(d)) * (d^k) = d^(k+1) under Dirichlet convolution,

        F_k(n) = P_{k+1}(n) - sum_{b=2..n} b^k F_k(floor(n/b)),

    and the b range is grouped by equal floor(n/b).
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.pref0, self.pref1, self.pref2 = self._build(limit)
        self.cache = {}

    @staticmethod
    def _build(limit: int):
        phi = array("I", range(limit + 1))

        for p in range(2, limit + 1):
            if phi[p] == p:
                for j in range(p, limit + 1, p):
                    phi[j] -= phi[j] // p

        pref0 = array("I", [0]) * (limit + 1)
        pref1 = array("I", [0]) * (limit + 1)
        pref2 = array("I", [0]) * (limit + 1)

        s0 = s1 = s2 = 0
        for i in range(1, limit + 1):
            ph = phi[i]
            im = i % MOD
            s0 = (s0 + ph) % MOD
            s1 = (s1 + im * ph) % MOD
            s2 = (s2 + im * im % MOD * ph) % MOD
            pref0[i] = s0
            pref1[i] = s1
            pref2[i] = s2

        return pref0, pref1, pref2

    def values(self, n: int):
        if n <= self.limit:
            return self.pref0[n], self.pref1[n], self.pref2[n]

        cached = self.cache.get(n)
        if cached is not None:
            return cached

        f0 = p1(n)
        f1 = p2(n)
        f2 = p3(n)

        l = 2
        while l <= n:
            q = n // l
            r = n // q

            sum_0 = (r - l + 1) % MOD
            sum_1 = (p1(r) - p1(l - 1)) % MOD
            sum_2 = (p2(r) - p2(l - 1)) % MOD

            sub0, sub1, sub2 = self.values(q)
            f0 = (f0 - sum_0 * sub0) % MOD
            f1 = (f1 - sum_1 * sub1) % MOD
            f2 = (f2 - sum_2 * sub2) % MOD

            l = r + 1

        out = f0, f1, f2
        self.cache[n] = out
        return out


def nonconcurrent_candidate_count(m: int, n: int) -> int:
    """
    Count all triples of segments which would form a triangle if no three of
    them were concurrent at one point.
    """
    two_same_bottom = (
        (m % MOD)
        * ((m - 1) % MOD)
        % MOD
        * (n % MOD)
        % MOD
        * ((n - 1) % MOD)
        % MOD
        * ((n + 1) % MOD)
        % MOD
        * INV6
        % MOD
    )

    distinct_bottoms = c3_mod(m) * ((c3_mod(n + 2) - n) % MOD) % MOD
    return (two_same_bottom + distinct_bottoms) % MOD


def weighted_gcd_sum(m: int, n: int, tp: TotientPrefix) -> int:
    """
    Sum gcd(a,b)*(m-a)*(n-b) for 1<=a<m and 1<=b<n, modulo MOD.
    """
    m1 = m - 1
    n1 = n - 1
    upper = min(m1, n1)
    total = 0

    l = 1
    while l <= upper:
        qm = m1 // l
        qn = n1 // l
        r = min(m1 // qm, n1 // qn, upper)

        r0, r1, r2 = tp.values(r)
        l0, l1, l2 = tp.values(l - 1)

        s0 = (r0 - l0) % MOD
        s1 = (r1 - l1) % MOD
        s2 = (r2 - l2) % MOD

        qm_mod = qm % MOD
        qn_mod = qn % MOD

        # A_x(d) = sum_{k=1..floor((x-1)/d)} (x-kd)
        #        = a0 + a1*d on this interval.
        a0m = qm_mod * (m % MOD) % MOD
        a1m = -qm_mod * ((qm + 1) % MOD) % MOD * INV2 % MOD
        a0n = qn_mod * (n % MOD) % MOD
        a1n = -qn_mod * ((qn + 1) % MOD) % MOD * INV2 % MOD

        c0 = a0m * a0n % MOD
        c1 = (a0m * a1n + a1m * a0n) % MOD
        c2 = a1m * a1n % MOD

        total = (total + c0 * s0 + c1 * s1 + c2 * s2) % MOD
        l = r + 1

    return total


def concurrent_triple_count(m: int, n: int, tp: TotientPrefix) -> int:
    """
    Count collinear negative-slope triples of grid points, equivalently triples
    of segments that are all concurrent at one interior point.
    """
    gcd_part = weighted_gcd_sum(m, n, tp)
    endpoint_pairs_without_interior_choice = c2_mod(m) * c2_mod(n) % MOD
    return (gcd_part - endpoint_pairs_without_interior_choice) % MOD


def T(m: int, n: int, tp: TotientPrefix) -> int:
    return (
        nonconcurrent_candidate_count(m, n)
        - concurrent_triple_count(m, n, tp)
    ) % MOD


def main() -> None:
    # Larger limits use more memory but reduce the amount of recursive summation.
    # Override, for example, with: SIEVE_LIMIT=5000000 python3 main.py
    sieve_limit = int(os.environ.get("SIEVE_LIMIT", "10000000"))
    tp = TotientPrefix(sieve_limit)

    assert T(2, 3, tp) == 8
    assert T(3, 5, tp) == 146
    assert T(12, 23, tp) == 756716

    print(T(1234 * 10**8, 2345 * 10**8, tp))


if __name__ == "__main__":
    main()
