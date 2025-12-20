#!/usr/bin/env python3
"""
Project Euler 794 — Seventeen Points

At step n we have chosen x1..xn in [0,1) so that each interval [k/n,(k+1)/n) contains
exactly one of these n points.

Key observation
---------------
For a fixed n, "one point per interval" means that after sorting the first n points,
the k-th smallest point must lie in [k/n,(k+1)/n). Therefore the whole procedure is
determined by *where the new point is inserted* in the sorted order at each step.

For any fixed insertion history, every point accumulates interval constraints from
all later steps. The feasible range for a point is the intersection of those
half-open intervals. To minimize the total sum we take each point as small as
possible, i.e. at the maximum of its lower bounds.

We depth-first search all feasible insertion histories up to n=18 (small enough),
track the minimal sum at n=17, and also confirm that no history reaches n=18
(as stated in the problem).

All arithmetic is exact:
every boundary k/n is represented as an integer scaled by D = lcm(1..max_n).

Running this file prints F(17) rounded to 12 decimal places.
"""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_UP, getcontext
from fractions import Fraction
from typing import List, Optional, Tuple


def lcm(a: int, b: int) -> int:
    return a // math.gcd(a, b) * b


def search_best_sum(max_n: int, target_n: int) -> Tuple[Optional[int], int, bool]:
    """
    Explore all feasible insertion histories up to 'max_n'.

    Returns:
        best_scaled: minimal sum at n=target_n, scaled by denom (or None if impossible)
        denom:       common denominator (lcm(1..max_n))
        exists_max:  True iff a valid construction reaches n=max_n
    """
    if not (1 <= target_n <= max_n):
        raise ValueError("Require 1 <= target_n <= max_n")

    # Common denominator so all k/n become integers.
    denom = 1
    for i in range(1, max_n + 1):
        denom = lcm(denom, i)

    scale = [0] * (max_n + 1)
    for n in range(1, max_n + 1):
        scale[n] = denom // n

    # Lower/upper bounds for each point id (1..max_n), scaled by denom.
    L = [0] * (max_n + 1)
    U = [denom] * (max_n + 1)

    # Current sorted order of point ids among the first n points.
    order: List[int] = [1]

    best_scaled: Optional[int] = None
    exists_max = False

    def rec(n: int, order: List[int], L: List[int], U: List[int], sumL: int) -> None:
        nonlocal best_scaled, exists_max

        # Record a candidate at the target depth.
        if n == target_n:
            if best_scaled is None or sumL < best_scaled:
                best_scaled = sumL

        if n == max_n:
            exists_max = True
            return

        m = n + 1
        sc = scale[m]

        # Try all insertion positions for the new point id m.
        for pos in range(m):
            new_order = order[:pos] + [m] + order[pos:]
            newL = L[:]  # small arrays (<= 19 elements), copying is cheap
            newU = U[:]
            new_sumL = sumL

            feasible = True
            for k, pid in enumerate(new_order):
                lb = k * sc
                ub = (k + 1) * sc

                if lb > newL[pid]:
                    new_sumL += lb - newL[pid]
                    newL[pid] = lb

                if ub < newU[pid]:
                    newU[pid] = ub

                if newL[pid] >= newU[pid]:
                    feasible = False
                    break

            if feasible:
                rec(m, new_order, newL, newU, new_sumL)

    rec(1, order, L, U, 0)
    return best_scaled, denom, exists_max


def format_scaled(value_scaled: int, denom: int, places: int = 12) -> str:
    """Convert value_scaled/denom to a decimal string rounded to 'places' decimals."""
    getcontext().prec = 80
    val = Decimal(value_scaled) / Decimal(denom)
    q = Decimal(1).scaleb(-places)  # 10^-places
    return str(val.quantize(q, rounding=ROUND_HALF_UP))


def main() -> None:
    # --- Asserts from the problem statement ---
    best4, d4, _ = search_best_sum(4, 4)
    assert best4 is not None
    assert Fraction(best4, d4) == Fraction(3, 2)  # F(4) = 1.5

    # Compute the target value while also checking that the construction
    # cannot be extended to 18 points (as stated in the problem).
    best17, denom, exists18 = search_best_sum(18, 17)
    assert exists18 is False
    assert best17 is not None  # guaranteed by the problem statement

    # --- Print the required answer ---
    print(format_scaled(best17, denom, 12))


if __name__ == "__main__":
    main()
