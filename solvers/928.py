#!/usr/bin/env python3
"""
Project Euler 928 - Cribbage

Count the number of non-empty Hands (subsets of a standard 52-card deck)
for which:

    Hand score == Cribbage score

Card values:
- Ace = 1
- 2..10 = pip value
- Jack/Queen/King = 10

Cribbage score (problem variant):
- Pairs: each unordered pair of same rank scores 2 points.
- Runs: consider maximal consecutive segments of ranks present in the hand.
        For each such segment with length >= 3, score:
            length * (product of multiplicities in that segment)
- Fifteens: each subset of cards whose VALUES sum to 15 scores 2 points.

No external libraries used.
"""

from itertools import product
from math import comb
from typing import Dict, Tuple


# --- small precomputations ---
COMB4 = (1, 4, 6, 4, 1)  # C(4, c)
CHOOSE = [[comb(n, k) for k in range(n + 1)] for n in range(5)]  # n<=4


def rank_value(rank: int) -> int:
    return 10 if rank >= 10 else rank


def dp_update(dp: Tuple[int, ...], v: int, c: int) -> Tuple[int, ...]:
    """
    Update subset-sum polynomial coefficients up to degree 15.

    dp[s] = number of ways to choose a subset from processed ranks summing to s
    (counting choices among distinct cards of identical rank via binomials).

    Multiply by sum_{k=0..c} C(c,k) x^(k*v).
    """
    if c == 0:
        return dp
    nd = [0] * 16
    cc = CHOOSE[c]
    for s in range(16):
        tot = 0
        maxk = min(c, s // v)
        for k in range(maxk + 1):
            tot += dp[s - k * v] * cc[k]
        nd[s] = tot
    return tuple(nd)


def score_from_counts(counts_1_to_13) -> Tuple[int, int]:
    """
    Given multiplicities per rank (index 1..13), return:
        (hand_score, cribbage_score)
    """
    counts = counts_1_to_13

    # Hand score + pair score
    hand = 0
    pairs = 0
    for r in range(1, 14):
        c = counts[r]
        hand += c * rank_value(r)
        pairs += c * (c - 1)  # equals 2*C(c,2)

    # Run score: maximal consecutive segments of positive counts
    run_score = 0
    cur_len = 0
    cur_prod = 1
    for r in range(1, 14):
        c = counts[r]
        if c == 0:
            if cur_len >= 3:
                run_score += cur_len * cur_prod
            cur_len = 0
            cur_prod = 1
        else:
            cur_len += 1
            cur_prod *= c
    if cur_len >= 3:
        run_score += cur_len * cur_prod

    # Fifteens via subset-sum DP up to 15
    dp = (1,) + (0,) * 15
    for r in range(1, 14):
        dp = dp_update(dp, rank_value(r), counts[r])
    fifteens = dp[15]
    fifteen_score = 2 * fifteens

    crib = pairs + run_score + fifteen_score
    return hand, crib


def _run_statement_asserts() -> None:
    # Example 1: (5♠, 5♣, 5♦, K♥) has Cribbage score 14
    c = [0] * 14
    c[5] = 3
    c[13] = 1  # King
    hs, cs = score_from_counts(c)
    assert cs == 14

    # Example 2: (A♦, A♥, 2♣, 3♥, 4♣, 5♠) has Cribbage score 16
    c = [0] * 14
    c[1] = 2
    c[2] = 1
    c[3] = 1
    c[4] = 1
    c[5] = 1
    hs, cs = score_from_counts(c)
    assert cs == 16
    assert hs == cs


def solve() -> int:
    """
    Meet-in-the-middle split at ranks 7/8.

    Left ranks: 1..7 (values 1..7)
    Right ranks: 8..13 (values 8,9,10,10,10,10)

    Key simplification:
    On the right side, subset sums <=15 can only be 0,8,9,10
    (since 8+9>15 and 10+anything>=18, 10+10>15).
    Therefore:
        total_fifteens = L15 + L7*R8 + L6*R9 + L5*R10
    where:
        R8 = number of ways to pick sum 8 from right = c8
        R9 = number of ways to pick sum 9 from right = c9
        R10 = number of ways to pick sum 10 from right = total number of 10-valued cards on right

    Runs are handled by keeping the trailing segment on the left (ending at rank 7)
    and the leading segment on the right (starting at rank 8) “pending” and
    combining them at the boundary.
    """
    _run_statement_asserts()

    # left: key -> {A_L: weight}
    # key = (L5, L6, L7, L15, tail_len, tail_prod)
    left: Dict[Tuple[int, int, int, int, int, int], Dict[int, int]] = {}

    # Enumerate left multiplicities: 5^7 = 78125
    for c1, c2, c3, c4, c5, c6, c7 in product(range(5), repeat=7):
        counts = (c1, c2, c3, c4, c5, c6, c7)

        # suit choices weight
        w = (
            COMB4[c1]
            * COMB4[c2]
            * COMB4[c3]
            * COMB4[c4]
            * COMB4[c5]
            * COMB4[c6]
            * COMB4[c7]
        )
        if w == 0:
            continue

        # hand value and pair score for left
        hv = 1 * c1 + 2 * c2 + 3 * c3 + 4 * c4 + 5 * c5 + 6 * c6 + 7 * c7
        pairs = (
            c1 * (c1 - 1)
            + c2 * (c2 - 1)
            + c3 * (c3 - 1)
            + c4 * (c4 - 1)
            + c5 * (c5 - 1)
            + c6 * (c6 - 1)
            + c7 * (c7 - 1)
        )

        # subset-sum DP up to 15 for ranks 1..7
        dp = (1,) + (0,) * 15
        dp = dp_update(dp, 1, c1)
        dp = dp_update(dp, 2, c2)
        dp = dp_update(dp, 3, c3)
        dp = dp_update(dp, 4, c4)
        dp = dp_update(dp, 5, c5)
        dp = dp_update(dp, 6, c6)
        dp = dp_update(dp, 7, c7)

        L5, L6, L7, L15 = dp[5], dp[6], dp[7], dp[15]

        # dp15_total >= L15, and final hand score <= 340 => dp15_total <= 170
        if L15 > 170:
            continue

        # run_internal on left excludes the trailing segment that reaches rank 7
        run_internal = 0
        cur_len = 0
        cur_prod = 1
        for c in counts:
            if c == 0:
                if cur_len >= 3:
                    run_internal += cur_len * cur_prod
                cur_len = 0
                cur_prod = 1
            else:
                cur_len += 1
                cur_prod *= c

        tail_len = cur_len
        tail_prod = cur_prod if tail_len > 0 else 1

        A_L = hv - pairs - run_internal

        key = (L5, L6, L7, L15, tail_len, tail_prod)
        d = left.get(key)
        if d is None:
            left[key] = {A_L: w}
        else:
            d[A_L] = d.get(A_L, 0) + w

    # right: grouped by (c8, c9, n10) then by (head_len, head_prod) then A_R
    right: Dict[Tuple[int, int, int], Dict[Tuple[int, int], Dict[int, int]]] = {}

    # Enumerate right multiplicities: 5^6 = 15625
    for c8, c9, c10, c11, c12, c13 in product(range(5), repeat=6):
        w = COMB4[c8] * COMB4[c9] * COMB4[c10] * COMB4[c11] * COMB4[c12] * COMB4[c13]
        if w == 0:
            continue

        n10 = c10 + c11 + c12 + c13
        hv = 8 * c8 + 9 * c9 + 10 * n10
        pairs = (
            c8 * (c8 - 1)
            + c9 * (c9 - 1)
            + c10 * (c10 - 1)
            + c11 * (c11 - 1)
            + c12 * (c12 - 1)
            + c13 * (c13 - 1)
        )

        # head segment (starting at rank 8) is excluded from run_internal
        counts = (c8, c9, c10, c11, c12, c13)
        head_len = 0
        head_prod = 1
        for c in counts:
            if c == 0:
                break
            head_len += 1
            head_prod *= c
        head_state = (head_len, head_prod) if head_len > 0 else (0, 1)

        # run_internal for right: count all maximal segments length>=3 EXCEPT
        # the one that starts at rank 8.
        run_internal = 0
        cur_len = 0
        cur_prod = 1
        seg_start = -1
        for i, c in enumerate(counts):
            if c == 0:
                if cur_len >= 3 and seg_start != 0:
                    run_internal += cur_len * cur_prod
                cur_len = 0
                cur_prod = 1
                seg_start = -1
            else:
                if cur_len == 0:
                    seg_start = i
                cur_len += 1
                cur_prod *= c
        if cur_len >= 3 and seg_start != 0:
            run_internal += cur_len * cur_prod

        A_R = hv - pairs - run_internal

        tkey = (c8, c9, n10)
        hdict = right.get(tkey)
        if hdict is None:
            right[tkey] = {head_state: {A_R: w}}
        else:
            amap = hdict.get(head_state)
            if amap is None:
                hdict[head_state] = {A_R: w}
            else:
                amap[A_R] = amap.get(A_R, 0) + w

    # Precompute run-boundary contributions for tail/head state pairs
    runb_cache: Dict[Tuple[Tuple[int, int], Tuple[int, int]], int] = {}

    def run_boundary(tail: Tuple[int, int], head: Tuple[int, int]) -> int:
        k = (tail, head)
        v = runb_cache.get(k)
        if v is not None:
            return v
        tl, tp = tail
        hl, hp = head
        if tl > 0 and hl > 0:
            L = tl + hl
            P = tp * hp
        elif tl > 0:
            L = tl
            P = tp
        elif hl > 0:
            L = hl
            P = hp
        else:
            runb_cache[k] = 0
            return 0
        res = (L * P) if L >= 3 else 0
        runb_cache[k] = res
        return res

    total = 0

    # Combine
    # Condition: (A_L + A_R - runb) == 2 * dp15_total
    # dp15_total = L15 + L7*c8 + L6*c9 + L5*n10
    for (L5, L6, L7, L15, tl, tp), left_amap in left.items():
        tail_state = (tl, tp)

        # small optimization: most left_amap have exactly one entry
        if len(left_amap) == 1:
            (aL, wL) = next(iter(left_amap.items()))

            for (c8, c9, n10), head_dict in right.items():
                dp15 = L15 + L7 * c8 + L6 * c9 + L5 * n10
                if dp15 > 170:
                    continue
                base = 2 * dp15

                for head_state, right_amap in head_dict.items():
                    req = base + run_boundary(tail_state, head_state)
                    want = req - aL
                    wR = right_amap.get(want)
                    if wR:
                        total += wL * wR
        else:
            for (c8, c9, n10), head_dict in right.items():
                dp15 = L15 + L7 * c8 + L6 * c9 + L5 * n10
                if dp15 > 170:
                    continue
                base = 2 * dp15

                for head_state, right_amap in head_dict.items():
                    req = base + run_boundary(tail_state, head_state)
                    # sum over aL: left[aL] * right[req-aL]
                    for aL, wL in left_amap.items():
                        wR = right_amap.get(req - aL)
                        if wR:
                            total += wL * wR

    # Exclude the empty hand (all multiplicities 0), which would satisfy 0==0
    # but is not allowed by the statement.
    total -= 1
    return total


def main() -> None:
    print(solve())


if __name__ == "__main__":
    main()
