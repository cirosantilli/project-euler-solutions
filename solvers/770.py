#!/usr/bin/env python3
"""
Project Euler 770 - Delphi Flip

The solution relies on turning the minimax game into a simple closed form:
    g(X) is the smallest n such that A can guarantee at least X grams.

We avoid any external libraries; only Python's standard library is used.
"""

from __future__ import annotations

import math


def _ln_p_central_binom_over_4n(n: int) -> float:
    """
    Return ln(p_n) where p_n = C(2n, n) / 4^n.

    For very large n, computing C(2n,n) is infeasible, so we use the
    Stirling-series expansion (derived from log-factorials):

        ln p_n = -1/2 * ln(pi*n) - 1/(8n) + 1/(192 n^3) - 1/(640 n^5) + O(n^-7)

    For the n that matter in this problem, the omitted O(n^-7) term is
    astronomically small.
    """
    nf = float(n)
    inv = 1.0 / nf
    inv2 = inv * inv
    inv3 = inv2 * inv
    inv5 = inv3 * inv2
    return (
        -0.5 * math.log(math.pi * nf)
        - 0.125 * inv
        + (1.0 / 192.0) * inv3
        - (1.0 / 640.0) * inv5
    )


def _p_leq_r_exact(n: int, r_num: int, r_den: int) -> bool:
    """
    Check whether p_n <= r exactly for modest n, where:
        p_n = C(2n, n) / 4^n
        r = r_num / r_den

    Uses integer arithmetic:
        C(2n,n) * r_den <= 4^n * r_num
        where 4^n = 2^(2n).
    """
    c = math.comb(2 * n, n)
    return c * r_den <= (1 << (2 * n)) * r_num


def g_for_fraction(x_num: int, x_den: int) -> int:
    """
    Compute g(X) for X = x_num/x_den (with X < 2).

    Using the closed form for the game's value:
        F(n) = guaranteed final gold starting from 1 gram
             = 2 / (1 + p_n),  where p_n = C(2n,n)/4^n.

    The condition F(n) >= X is equivalent to:
        p_n <= (2 - X) / X.

    We compute the minimal such n.

    Strategy:
    - If the estimated n is small, test n incrementally with exact integer comparisons.
    - If n is huge, compare using logs of p_n via the Stirling expansion.
    """
    if x_den <= 0:
        raise ValueError("Denominator must be positive.")
    if x_num <= 0:
        raise ValueError("X must be positive.")
    if x_num >= 2 * x_den:
        raise ValueError(
            "X must be < 2 for this game (the guarantee approaches 2 from below)."
        )

    # r = (2 - X) / X = (2*x_den - x_num) / x_num
    r_num = 2 * x_den - x_num
    r_den = x_num
    g = math.gcd(r_num, r_den)
    r_num //= g
    r_den //= g

    # Quick estimate from p_n ~ 1/sqrt(pi*n) => n ~ 1/(pi*r^2)
    r = r_num / r_den
    if r == 0.0:
        return 0  # Only possible when X=2, which we forbid above.

    n_est = int(1.0 / (math.pi * r * r))
    if n_est < 20000:
        # Exact incremental search is fast here.
        n = 0
        while not _p_leq_r_exact(n, r_num, r_den):
            n += 1
        return n

    # Huge n: compare in log-space using Stirling expansion.
    ln_r = math.log(r_num) - math.log(r_den)

    # Start near the estimate and adjust locally (should take only a few steps).
    n = max(1, n_est - 10)
    while _ln_p_central_binom_over_4n(n) > ln_r:
        n += 1
    while n > 1 and _ln_p_central_binom_over_4n(n - 1) <= ln_r:
        n -= 1
    return n


def main() -> None:
    # Test value from the problem statement:
    assert g_for_fraction(17, 10) == 10  # g(1.7) = 10

    # Compute and print g(1.9999)
    ans = g_for_fraction(19999, 10000)
    print(ans)


if __name__ == "__main__":
    main()
