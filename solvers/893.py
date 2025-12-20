#!/usr/bin/env python3
"""
Project Euler 893: Matchsticks

Compute T(10^6) = sum_{n=1..10^6} M(n), where M(n) is the minimum number of matchsticks
needed to display an expression (using decimal digits, +, × with standard precedence)
that evaluates to n.

Constraints:
- No external libraries (standard library only).
- Include asserts for test values given in the problem statement.
- Do NOT hardcode the final answer; print it.
"""

from __future__ import annotations

import math

# 7-segment matchstick counts for digits 0..9, per the problem statement diagram.
_DIGIT_COST = (6, 2, 5, 5, 4, 5, 6, 3, 7, 6)

# Each operator uses 2 matchsticks.
_OP_COST = 2


def _digit_costs_upto(n_max: int) -> bytearray:
    """digit_cost[n] = matchsticks to display decimal n (no leading zeros)."""
    dc = bytearray(n_max + 1)
    for n in range(1, n_max + 1):
        dc[n] = dc[n // 10] + _DIGIT_COST[n % 10]
    return dc


def _spf_sieve(n_max: int) -> list[int]:
    """Smallest-prime-factor sieve."""
    spf = list(range(n_max + 1))
    limit = int(n_max**0.5)
    for i in range(2, limit + 1):
        if spf[i] == i:  # prime
            start = i * i
            step = i
            for j in range(start, n_max + 1, step):
                if spf[j] == j:
                    spf[j] = i
    return spf


def _compute_P(n_max: int) -> bytearray:
    """
    P(n): minimum matchsticks to display n using only decimal digits and multiplication.

    P(n) = min( digit_cost(n),
                min_{d|n, 2<=d<=sqrt(n)} P(d) + P(n/d) + 2 )
    """
    digit_cost = _digit_costs_upto(n_max)
    spf = _spf_sieve(n_max)

    P = bytearray(digit_cost)
    P[0] = 255  # sentinel (unused)

    for n in range(2, n_max + 1):
        if spf[n] == n:
            continue  # prime: digit form is best among multiplicative-only

        # Factorize n using spf.
        m = n
        factors: list[tuple[int, int]] = []
        while m > 1:
            p = spf[m]
            e = 0
            while m % p == 0:
                m //= p
                e += 1
            factors.append((p, e))

        # Generate divisors from prime powers.
        divs = [1]
        for p, e in factors:
            prev = divs
            divs = []
            pe = 1
            for _ in range(e + 1):
                for d in prev:
                    divs.append(d * pe)
                pe *= p

        r = math.isqrt(n)
        best = P[n]
        for d in divs:
            if 1 < d <= r:
                q = n // d
                cand = P[d] + P[q] + _OP_COST
                if cand < best:
                    best = cand
        P[n] = best

    return P


def _best_two_term_psum(n_max: int, P: bytearray, psum_limit: int) -> bytearray:
    """
    best2[s] = min_{a+b=s} (P[a]+P[b]) if that minimum <= psum_limit, else 255.

    Computed by bucketing numbers by P-cost and enumerating cost-pairs with c1+c2<=psum_limit.
    """
    L = psum_limit
    buckets: list[list[int]] = [[] for _ in range(L + 1)]
    for x in range(1, n_max + 1):
        c = P[x]
        if c <= L:
            buckets[c].append(x)

    best2 = bytearray([255]) * (n_max + 1)

    N = n_max
    for c1 in range(2, L + 1):
        A = buckets[c1]
        if not A:
            continue
        max_c2 = L - c1
        for c2 in range(c1, max_c2 + 1):
            B = buckets[c2]
            if not B:
                continue
            psum = c1 + c2

            if c1 == c2:
                Ls = A
                Llen = len(Ls)
                for i, a in enumerate(Ls):
                    max_b = N - a
                    hi = Llen - 1
                    while hi >= i and Ls[hi] > max_b:
                        hi -= 1
                    for j in range(i, hi + 1):
                        s = a + Ls[j]
                        if psum < best2[s]:
                            best2[s] = psum
            else:
                # Iterate outer over smaller list to reduce Python overhead.
                if len(A) <= len(B):
                    outer, inner = A, B
                else:
                    outer, inner = B, A

                inner_list = inner
                hi = len(inner_list) - 1
                for a in outer:
                    max_b = N - a
                    while hi >= 0 and inner_list[hi] > max_b:
                        hi -= 1
                    for j in range(hi + 1):
                        s = a + inner_list[j]
                        if psum < best2[s]:
                            best2[s] = psum

    return best2


def _compute_M_two_terms(n_max: int, P: bytearray) -> bytearray:
    """
    Two-term optimum:
      M2(n) = min( P(n), min_{a+b=n} (P(a)+P(b)+2) )

    We compute this exactly using:
    - bucket enumeration for all pairs yielding cost <= 34
    - and a fallback scan over all a with P(a) <= 18 for the remaining higher-cost cases
    """
    M = bytearray(P)
    buckets: list[list[int]] = [[] for _ in range(41)]
    for x in range(1, n_max + 1):
        buckets[P[x]].append(x)

    N = n_max
    M_arr = M

    # Strong improvements (final cost <= 34) by enumerating P-cost pairs with sum <= 32
    for c1 in range(2, 33):
        A = buckets[c1]
        if not A:
            continue
        max_c2 = 32 - c1
        for c2 in range(c1, max_c2 + 1):
            B = buckets[c2]
            if not B:
                continue
            cand_cost = c1 + c2 + _OP_COST  # <= 34

            if c1 == c2:
                Ls = A
                Llen = len(Ls)
                for i, a in enumerate(Ls):
                    max_b = N - a
                    hi = Llen - 1
                    while hi >= i and Ls[hi] > max_b:
                        hi -= 1
                    for j in range(i, hi + 1):
                        s = a + Ls[j]
                        if cand_cost < M_arr[s]:
                            M_arr[s] = cand_cost
            else:
                if len(A) <= len(B):
                    outer, inner = A, B
                else:
                    outer, inner = B, A

                inner_list = inner
                hi = len(inner_list) - 1
                for a in outer:
                    max_b = N - a
                    while hi >= 0 and inner_list[hi] > max_b:
                        hi -= 1
                    for j in range(hi + 1):
                        s = a + inner_list[j]
                        if cand_cost < M_arr[s]:
                            M_arr[s] = cand_cost

    # Finish exactness for high costs by scanning low P(a) (<=18).
    low_cost_summands: list[int] = []
    for c in range(2, 19):
        low_cost_summands.extend(buckets[c])
    low_cost_summands.sort()

    for n in range(1, N + 1):
        if M_arr[n] > 34:
            best = M_arr[n]
            half = n // 2
            for a in low_cost_summands:
                if a > half:
                    break
                b = n - a
                cand = P[a] + P[b] + _OP_COST
                if cand < best:
                    best = cand
            M_arr[n] = best

    return M_arr


def _apply_three_term_fix(n_max: int, P: bytearray, M: bytearray) -> None:
    """
    Allow 3-term sums:
      a + b + c  with cost  P(a)+P(b)+P(c)+4

    We compute best2[r] = min_{x+y=r} P(x)+P(y) for small sums, then test n = r + c.
    This targeted pass is enough to capture the rare case(s) where 3 additive terms
    beat all 1- or 2-term expressions.
    """
    # Any useful 3-term improvement for n<=1e6 happens with a small two-term P-sum.
    best2 = _best_two_term_psum(n_max, P, psum_limit=30)

    # "Cheap" third terms.
    cheap_c: list[int] = []
    for x in range(1, n_max + 1):
        if P[x] <= 20:
            cheap_c.append(x)
    cheap_c.sort()

    # Tiny remainders r where best2[r] is extremely small.
    small_r: list[int] = []
    for r in range(2, n_max + 1):
        v = best2[r]
        if v != 255 and v <= 8:
            small_r.append(r)
    small_r.sort()

    for n in range(1, n_max + 1):
        cur = M[n]
        if cur < 33:
            continue

        target = cur - 1
        psum_limit = target - 4  # need P(a)+P(b)+P(c) <= psum_limit
        if psum_limit < 6:  # min is 2+2+2
            continue

        best = cur

        # Case 1: cheap c
        for c in cheap_c:
            if c >= n:
                break
            pc = P[c]
            if pc > psum_limit:
                continue
            r = n - c
            v = best2[r]
            if v != 255 and v + pc <= psum_limit:
                cand = v + pc + 4
                if cand < best:
                    best = cand
                    if best == target:
                        break

        # Case 2: expensive c but tiny remainder r
        if best > target:
            for r in small_r:
                if r >= n:
                    break
                v = best2[r]
                c = n - r
                pc = P[c]
                if v + pc <= psum_limit:
                    cand = v + pc + 4
                    if cand < best:
                        best = cand
                        if best == target:
                            break

        M[n] = best


def main() -> None:
    N = 1_000_000
    P = _compute_P(N)

    # Exact optimum among 1-term and 2-term sums
    M = _compute_M_two_terms(N, P)

    # Apply the needed 3-term improvement pass
    _apply_three_term_fix(N, P, M)

    # Asserts for values given in the problem statement:
    assert M[28] == 9
    assert sum(M[1:101]) == 916

    print(sum(M[1:]))


if __name__ == "__main__":
    main()
