#!/usr/bin/env python3
"""
Project Euler 782: Distinct Rows and Columns

We need C(n) = sum_{k=0..n^2} c(n,k) where c(n,k) is the minimum possible
"complexity" (number of distinct rows and columns) among n×n binary matrices
with exactly k ones.

This solver is self-contained (standard library only).
"""

from __future__ import annotations

from math import isqrt


def _count_k_complexity_le_3(n: int) -> int:
    """
    Return |S3|, the number of k in [0..n^2] for which there exists an n×n
    matrix with exactly k ones and complexity <= 3.

    Key fact used:
      Every complexity <= 3 matrix can be permuted into a 3×3 block "blow-up"
      of a 3×3 binary template that has at most 3 distinct row/column patterns.
      Up to permutations of the 3 groups and complementing (k -> n^2-k),
      all such templates fall into 7 orbits, which yield 7 canonical quadratic
      forms in the group sizes.

    We mark all attainable k via those 7 forms and use complement symmetry by
    mapping k -> min(k, n^2-k) into [0..floor(n^2/2)].
    """
    n2 = n * n
    half = n2 // 2
    seen = bytearray(half + 1)
    seen0 = seen  # local alias for speed

    # --- Orbit 1: k = x*y with 0 <= x <= y <= n  (multiplication table triangle)
    # We mark values min(x*y, n^2 - x*y) using two arithmetic progressions per y.
    for y in range(1, n + 1):
        # progression 1: multiples of y from 0..min(y^2, half)
        y2 = y * y
        end1 = y2 if y2 <= half else half
        # slice assignment is done in C and is much faster than a Python loop
        seen0[0 : end1 + 1 : y] = b"\x01" * (end1 // y + 1)

        # progression 2: for those multiples > half, map to n^2 - (y*x)
        if y2 > half:
            x0 = half // y + 1  # smallest x with y*x > half
            if x0 <= y:
                start = n2 - y2  # corresponds to x=y
                end = n2 - y * x0  # corresponds to x=x0
                # both start and end are within [0..half]
                seen0[start : end + 1 : y] = b"\x01" * ((end - start) // y + 1)

    # Always attainable: k=0 and k=n^2 -> mapped to 0
    seen0[0] = 1

    # --- Orbit 2: k = v^2 - d^2 for 0 <= d <= v <= n
    # (equivalently k = c(2v-c) from the template family).
    for v in range(0, n + 1):
        vv = v * v
        if vv <= half:
            dsq = 0
            step = 1
            for _ in range(v + 1):
                seen0[vv - dsq] = 1
                dsq += step
                step += 2
        else:
            # split by threshold where k crosses half:
            # vv - d^2 <= half  <=>  d^2 >= vv - half
            diff = vv - half
            d_start = isqrt(diff - 1) + 1  # minimal d with d^2 >= diff
            offset = n2 - vv
            dsq = 0
            step = 1
            for _ in range(d_start):
                # k > half, store complement: n^2 - (vv - d^2) = (n^2 - vv) + d^2
                seen0[offset + dsq] = 1
                dsq += step
                step += 2
            for _ in range(d_start, v + 1):
                # k <= half
                seen0[vv - dsq] = 1
                dsq += step
                step += 2

    # Precompute q(b) = 2*b*(n-b) for b=0..floor(n/2) (symmetry in b)
    bmax_global = n // 2
    q = [0] * (bmax_global + 1)
    # q[0]=0; update incrementally: q(b+1)-q(b)=2*(n-2b-1)
    cur = 0
    delta = 2 * (n - 1)
    for b in range(0, bmax_global):
        q[b] = cur
        cur += delta
        delta -= 4
    q[bmax_global] = cur

    # --- Orbit 3/4/5/6: depend only on s=a+b and ab=a(s-a),
    # using distinct ab values (symmetry a <-> b) to halve work.
    #
    # For fixed s:
    #   c = n - s
    #   ab = a(s-a) has values:
    #     s even  (s=2v):   ab = v^2 - d^2
    #     s odd   (s=2v+1): ab = v(v+1) - d(d+1)
    #
    # Then the four quadratic forms are:
    #   k3 = c^2 + 2ab
    #   k5 = c(2n-c) + 2ab
    #   k6 = c*s + ab
    #   k7 = 2*(c*s + ab)
    for s in range(0, n + 1):
        c = n - s
        c2 = c * c
        c2n = c * (2 * n - c)
        cs = c * s

        if s & 1 == 0:
            v = s // 2
            base = v * v
            dsq = 0
            step = 1
            for _ in range(v + 1):
                ab = base - dsq
                two_ab = ab << 1

                k = c2 + two_ab
                if k > half:
                    k = n2 - k
                seen0[k] = 1

                k = c2n + two_ab
                if k > half:
                    k = n2 - k
                seen0[k] = 1

                k = cs + ab
                if k > half:
                    k = n2 - k
                seen0[k] = 1

                k = (cs + ab) << 1
                if k > half:
                    k = n2 - k
                seen0[k] = 1

                dsq += step
                step += 2
        else:
            v = s // 2  # floor
            base = v * (v + 1)  # v(v+1)
            pr = 0  # d(d+1)
            step = 2  # pr(d+1)-pr(d) starts at 2
            for _ in range(v + 1):
                ab = base - pr
                two_ab = ab << 1

                k = c2 + two_ab
                if k > half:
                    k = n2 - k
                seen0[k] = 1

                k = c2n + two_ab
                if k > half:
                    k = n2 - k
                seen0[k] = 1

                k = cs + ab
                if k > half:
                    k = n2 - k
                seen0[k] = 1

                k = (cs + ab) << 1
                if k > half:
                    k = n2 - k
                seen0[k] = 1

                pr += step
                step += 2

    # --- Orbit 7: k = c^2 + 2*b*(n-b), with 0 <= b <= n-c and symmetry b<->n-b.
    for c in range(0, n + 1):
        c2 = c * c
        bmax = n - c
        if bmax > bmax_global:
            bmax = bmax_global
        for b in range(0, bmax + 1):
            k = c2 + q[b]
            if k > half:
                k = n2 - k
            seen0[k] = 1

    count_half = sum(seen0)
    if n2 & 1:
        return 2 * count_half
    # n^2 even: midpoint maps to itself and must not be double-counted
    return 2 * count_half - seen0[half]


def _count_k_complexity_eq_2(n: int) -> int:
    """
    Return N2 = number of k in [0..n^2] whose *minimum* complexity is exactly 2.

    Complexity 2 is achievable exactly for these shapes (up to complement):
      - a×a block of ones (and the complement): k = a^2, n^2-a^2
      - a×(n-a) rectangle duplicated symmetrically (and complement):
        k = 2a(n-a), n^2-2a(n-a)

    We exclude k=0 and k=n^2 (complexity 1).
    """
    n2 = n * n
    k2 = set()
    for a in range(0, n + 1):
        a2 = a * a
        t = 2 * a * (n - a)
        k2.add(a2)
        k2.add(n2 - a2)
        k2.add(t)
        k2.add(n2 - t)
    k2.discard(0)
    k2.discard(n2)
    return len(k2)


def C(n: int) -> int:
    """
    Compute C(n) = sum_{k=0..n^2} c(n,k).

    For n>=2:
      - c(n,0)=c(n,n^2)=1
      - c(n,k) is in {2,3,4} otherwise.
      - Let N2 be number of k with minimum complexity 2.
      - Let N4 be number of k with minimum complexity 4.
        (Equivalently, k not achievable with complexity <=3.)
      Then:
        C(n) = 3*(n^2+1) - 4 - N2 + N4
    """
    if n <= 1:
        # n=0 is degenerate; n=1 has k=0,1 with complexity 1 each.
        return 2 if n == 1 else 0

    n2 = n * n
    total = n2 + 1

    s3 = _count_k_complexity_le_3(n)  # values achievable with complexity <= 3
    n4 = total - s3
    n2cnt = _count_k_complexity_eq_2(n)

    return 3 * total - 4 - n2cnt + n4


def _self_test() -> None:
    # From the problem statement
    assert C(2) == 8
    assert C(5) == 64
    assert C(10) == 274
    assert C(20) == 1150


def main() -> None:
    _self_test()
    print(C(10_000))


if __name__ == "__main__":
    main()
