#!/usr/bin/env python3
"""
Project Euler 950: Pirate Treasure

This program computes
    sum_{k=1..6} T(10^16, 10^k+1, 1/sqrt(10^k+1))
and prints the last 9 digits.

We work only with p of the form p = 1/sqrt(D), where D is a positive non-square integer.
All required instances in the statement (and the final query) have this form.
No floating point arithmetic is used.
"""

from __future__ import annotations

from math import isqrt


MOD_9 = 10**9


def floor_div_sqrt(d: int, D: int) -> int:
    """
    Return floor(d / sqrt(D)) for integers d >= 0, D >= 1, using integer arithmetic.

    We compute an initial guess via isqrt(floor(d^2 / D)) and then fix it by checking
    t^2 * D <= d^2.
    """
    if d <= 0:
        return 0
    dd = d * d
    t = isqrt(dd // D)
    # Correct potential off-by-one errors due to the intermediate floor.
    while (t + 1) * (t + 1) * D <= dd:
        t += 1
    while t * t * D > dd:
        t -= 1
    return t


def ceil_div_sqrt(d: int, D: int) -> int:
    """
    Return ceil(d / sqrt(D)) for integers d >= 0 and non-square D.

    For d>0 and irrational sqrt(D), d/sqrt(D) is never an integer, so:
        ceil(d/sqrt(D)) = floor(d/sqrt(D)) + 1
    """
    if d <= 0:
        return 0
    return floor_div_sqrt(d, D) + 1


def initial_prefix_sum(N: int, C: int) -> int:
    """
    For p < 1 (true for p=1/sqrt(D) with D>=2), the most senior pirate survives
    for n = 1..(2C+2) with w(n)=0 and
        c(n) = max(C - floor((n-1)/2), 0).

    This function returns sum_{n=1..min(N,2C+2)} c(n).
    """
    if N <= 0:
        return 0
    limit = 2 * C + 2
    M = min(N, limit)
    # Last two terms in the range are zeros, so clamp to 2C.
    M = min(M, 2 * C)
    if M <= 0:
        return 0

    m = M // 2  # number of full pairs (2 terms per pair)
    # For j=0..m-1, values are (C-j) twice.
    s = 2 * (m * C - (m * (m - 1)) // 2)
    if M % 2 == 1:
        # Extra term at n=2m+1 is C-m.
        s += C - m
    return s


def next_reset(L: int, C: int, D: int) -> int:
    """
    Let L be a position where the most senior pirate survives (w(L)=0).
    The next position x>L where the most senior pirate survives again can be found
    from the feasibility inequality:
        2(x-L) + 2*floor(C / ceil((x-L)/sqrt(D))) >= x

    We iterate over the distinct values of y = floor(C / k) (there are O(sqrt(C)) of them).
    For a fixed y, the smallest x satisfying the inequality is x = 2L - 2y.
    We check whether this x is consistent with y.
    """
    if C == 0:
        # With zero coins, after L the next time the senior survives is at x=2L.
        return 2 * L

    t = 1
    while t <= C:
        y = C // t  # candidate value of floor(C / k)
        x = 2 * L - 2 * y
        d = x - L
        if d > 0:
            s = ceil_div_sqrt(d, D)
            if C // s == y:
                return x
        # Jump to the next t that changes C//t.
        t = (C // y) + 1

    # y = 0 case (k > C)
    x = 2 * L
    return x


def T(N: int, C: int, D: int) -> int:
    """
    Compute T(N, C, 1/sqrt(D)) = sum_{n=1..N} (c(n) + w(n)) exactly.

    After n = 2C+2, the process has long "death cascades":
    between two positions where the most senior pirate survives, the most senior
    pirate dies every time, so c(n) stays constant and w(n) grows by 1 each step.
    """
    if N <= 0:
        return 0

    # Initial region where the most senior pirate always survives and w(n)=0.
    start_reset = 2 * C + 2
    if N <= start_reset:
        return initial_prefix_sum(N, C)

    total = initial_prefix_sum(
        start_reset, C
    )  # includes n=start_reset (which contributes 0)
    L = start_reset
    cL = 0  # at n=2C+2, c=0

    while L < N:
        x = next_reset(L, C, D)
        if x > N:
            # Truncated final cascade: n = L+1..N
            d = N - L + 1  # number of terms from L to N inclusive
            # Sum_{k=1..d-1} (cL + k)
            total += (d - 1) * cL + (d - 1) * d // 2
            break

        d = x - L  # distance to next reset
        if d > 1:
            # n = L+1..x-1 contributes (cL + 1) .. (cL + d-1)
            total += (d - 1) * cL + (d - 1) * d // 2

        # At n=x the most senior pirate survives again. Compute c(x).
        required_votes = (x + 1) // 2  # ceil(x/2)
        free_votes = d  # the senior + (d-1) pirates who would die if he dies
        need_bribes = required_votes - free_votes
        if need_bribes < 0:
            need_bribes = 0

        s = ceil_div_sqrt(d, D)
        cost = need_bribes * s
        # By construction x is feasible, so cost <= C.
        cL = C - cost

        total += cL  # w(x)=0
        L = x

    return total


def solve() -> int:
    # Statement checks
    assert T(30, 3, 3) == 190
    assert T(50, 3, 31) == 385
    assert T(10**3, 101, 101) == 142427

    N = 10**16
    acc = 0
    for k in range(1, 7):
        C = 10**k + 1
        acc += T(N, C, C)

    return acc % MOD_9


if __name__ == "__main__":
    ans = solve()
    # Print the last 9 digits, padding with leading zeros if needed.
    print(f"{ans:09d}")
