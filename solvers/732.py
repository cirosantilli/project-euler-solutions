#!/usr/bin/env python3
"""Project Euler 732 - Standing on the Shoulders of Trolls

Compute Q(N): maximum total IQ of trolls that can escape.

No external libraries are used (only Python standard library).
"""

from __future__ import annotations

import math
from typing import List, Tuple


MOD = 1_000_000_007


def generate_trolls(n: int) -> List[Tuple[int, int, int]]:
    """Return list of (h, l, q) for trolls 0..n-1."""
    # Need r_0..r_{3n-1}
    total_r = 3 * n
    r = [0] * total_r

    p = 1  # 5^0 mod MOD
    for i in range(total_r):
        r[i] = (p % 101) + 50
        p = (p * 5) % MOD

    trolls = []
    for k in range(n):
        h = r[3 * k]
        l = r[3 * k + 1]
        q = r[3 * k + 2]
        trolls.append((h, l, q))
    return trolls


def ceil_div_sqrt2(s: int) -> int:
    """Return ceil(s / sqrt(2)) exactly using integer arithmetic.

    We need the smallest y such that y >= s/sqrt(2), equivalently
    sqrt(2)*y >= s, or 2*y^2 >= s^2.

    Let A = s^2. Then y = ceil(sqrt(A/2)).
    A direct exact formula is:
        y = isqrt((A-1)//2) + 1
    """
    a = s * s
    return math.isqrt((a - 1) // 2) + 1


def Q(n: int) -> int:
    """Maximum total IQ of escaping trolls for the instance size n."""
    trolls = generate_trolls(n)

    total_h = sum(h for h, _, _ in trolls)
    y = ceil_div_sqrt2(total_h)  # ceil(D_N) where D_N = total_h / sqrt(2)
    base = total_h - y

    # Transform to scheduling with processing times and deadlines:
    # If W is the cumulative removed shoulder-height so far,
    # troll i can escape when remaining height S = total_h - W satisfies
    #   S + l_i >= y
    # i.e.
    #   W <= total_h - (y - l_i) = base + l_i  (start constraint).
    # This is equivalent (for integer W) to the completion constraint:
    #   W + h_i <= base + l_i + h_i.
    # Therefore each troll corresponds to a job:
    #   processing time p = h_i
    #   profit = q_i
    #   deadline d = base + l_i + h_i
    jobs: List[Tuple[int, int, int]] = []  # (deadline, processing_time, profit)
    max_d = 0
    for h, l, q in trolls:
        d = base + l + h
        if h <= d:  # otherwise impossible to schedule at all
            jobs.append((d, h, q))
            if d > max_d:
                max_d = d

    jobs.sort()  # earliest deadline first

    # 0/1 knapsack DP over time (total removed height), limited by deadlines.
    # dp[t] = max profit with total processing time exactly t; -1 means impossible.
    dp = [-1] * (max_d + 1)
    dp[0] = 0

    for d, p, profit in jobs:
        for t in range(d, p - 1, -1):
            prev = dp[t - p]
            if prev != -1:
                cand = prev + profit
                if cand > dp[t]:
                    dp[t] = cand

    return max(dp)


def main() -> None:
    # Test values from the problem statement
    assert Q(5) == 401
    assert Q(15) == 941

    print(Q(1000))


if __name__ == "__main__":
    main()
