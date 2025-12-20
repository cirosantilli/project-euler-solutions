#!/usr/bin/env python3
"""
Project Euler 821: 123-Separable

A set S of positive integers is "123-separable" when S, 2S and 3S are pairwise disjoint.
Define F(n) as the maximum size of (S ∪ 2S ∪ 3S) ∩ {1,2,...,n} over all such sets S.

This program computes F(10^16) using only integer arithmetic and the Python standard library.
"""

from __future__ import annotations


def coprime_to_6_count(m: int) -> int:
    """Count integers in [1..m] that are not divisible by 2 or 3."""
    if m <= 0:
        return 0
    # Inclusion–exclusion for multiples of 2 and 3.
    return m - m // 2 - m // 3 + m // 6


def smooth_2_3_upto(limit: int) -> list[int]:
    """All numbers of the form 2^a * 3^b <= limit, sorted."""
    res: list[int] = []
    p2 = 1
    while p2 <= limit:
        p3 = 1
        while p2 * p3 <= limit:
            res.append(p2 * p3)
            p3 *= 3
        p2 *= 2
    res.sort()
    return res


def holes_upto(limit: int) -> list[int]:
    """
    For the 2^a 3^b subproblem, there is an almost-perfect packing where the uncovered
    points fall in a very sparse, explicit list. Up to a given limit, generate those holes.
    """
    holes: list[int] = []
    for x in (6, 24, 54):
        if x <= limit:
            holes.append(x)

    # Terms of the form 3 * 2^(3t+1) for t >= 2: 384, 3072, 24576, ...
    x = 384
    while x <= limit:
        holes.append(x)
        x *= 8

    # Terms of the form 3^(3t+2) for t >= 1: 243, 6561, 177147, ...
    x = 243
    while x <= limit:
        holes.append(x)
        x *= 27

    holes.sort()
    return holes


def best_covered_small(limit: int) -> int:
    """
    Exact solver for the reduced (k=1) subproblem when limit is small.

    We only need this for limit <= 48. The candidate anchors are exactly the 2^a 3^b <= limit.
    Enumerate all subsets of anchors and keep the best disjoint (S,2S,3S) arrangement.
    """
    vals = smooth_2_3_upto(limit)
    m = len(vals)
    doubles = [2 * v for v in vals]
    triples = [3 * v for v in vals]

    best = 0
    for mask in range(1 << m):
        S = set()
        D = set()
        T = set()
        ok = True
        for i, v in enumerate(vals):
            if (mask >> i) & 1:
                dv = doubles[i]
                tv = triples[i]
                # Enforce pairwise disjointness of S, 2S, 3S.
                if v in D or v in T or dv in S or dv in T or tv in S or tv in D:
                    ok = False
                    break
                S.add(v)
                D.add(dv)
                T.add(tv)
        if not ok:
            continue
        union = S | D | T
        covered = 0
        for u in union:
            if u <= limit:
                covered += 1
        if covered > best:
            best = covered
    return best


def F(n: int) -> int:
    """
    Compute F(n) exactly.

    Decomposition:
      Every x can be written uniquely as x = 2^a * 3^b * k with gcd(k,6)=1.
      Multiplying by 2 or 3 preserves k, so each k gives an independent copy of the
      2^a 3^b problem with limit floor(n/k).

    Let H(N) be the maximum covered count inside the set {2^a 3^b <= N}. Then:
      F(n) = sum_{k <= n, gcd(k,6)=1} H(floor(n/k)).

    H(N) is constant between consecutive 2^a 3^b values, so we sum over those ranges.
    """
    smooth = smooth_2_3_upto(n)
    idx = {v: i for i, v in enumerate(smooth)}

    # Precompute exact H for small limits (only needed up to 48).
    small_H: dict[int, int] = {}
    for v in smooth:
        if v > 48:
            break
        small_H[v] = best_covered_small(v)

    # Precompute hole prefix counts on smooth breakpoints.
    hole_list = holes_upto(n)
    hole_prefix: dict[int, int] = {}
    j = 0
    cnt = 0
    for v in smooth:
        while j < len(hole_list) and hole_list[j] <= v:
            cnt += 1
            j += 1
        hole_prefix[v] = cnt

    def H_at_breakpoint(v: int) -> int:
        if v <= 48:
            return small_H[v]
        # Total number of 2^a 3^b values <= v equals its index+1 in the sorted smooth list.
        total = idx[v] + 1
        return total - hole_prefix[v]

    ans = 0
    for i, L in enumerate(smooth):
        R = n if i + 1 == len(smooth) else min(n, smooth[i + 1] - 1)

        # floor(n/k) in [L, R]  <=>  n/(R+1) < k <= n/L
        k_low = n // (R + 1) + 1
        k_high = n // L
        if k_low > k_high:
            continue

        count_k = coprime_to_6_count(k_high) - coprime_to_6_count(k_low - 1)
        ans += H_at_breakpoint(L) * count_k

    return ans


def main() -> None:
    # Tests from the problem statement.
    assert F(6) == 5
    assert F(20) == 19

    n = 10**16
    print(F(n))


if __name__ == "__main__":
    main()
