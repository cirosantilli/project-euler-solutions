#!/usr/bin/env python3
"""Project Euler 930 - The Gathering

We model the process on relative positions of the balls.
The expected number of moves until all balls are in one bowl can be computed
from the spectrum of a translation-invariant random walk on (Z_n)^(m-1).

This file prints G(12, 12) in scientific notation with 12 digits after the
decimal point.

No external libraries are used.
"""

from __future__ import annotations

import math
from functools import lru_cache


# Precompute binomial coefficients up to 11 (because m <= 12 => d=m-1 <= 11).
_MAX_D = 11
_BINOM = [[0] * (_MAX_D + 1) for _ in range(_MAX_D + 1)]
for n in range(_MAX_D + 1):
    for k in range(n + 1):
        _BINOM[n][k] = math.comb(n, k)


def _format_sci(x: float) -> str:
    """Format like '1.681521567954e4' (no + sign, no leading zeros)."""
    s = f"{x:.12e}"  # e.g. 1.234567890123e+04
    mant, exp = s.split("e")
    return f"{mant}e{int(exp)}"


@lru_cache(maxsize=None)
def F(n: int, m: int) -> float:
    """Expected number of moves until all m balls are in the same bowl.

    The expectation is over both:
      * the random initial placement (iid uniform over bowls), and
      * the random evolution steps.

    Computed via: F = sum_{k != 0} 1/(1 - lambda_k), where lambda_k are the
    eigenvalues of the induced random walk on (Z_n)^(m-1).
    """
    if n < 2 or m < 2:
        raise ValueError("n and m must be at least 2")

    d = m - 1  # dimension
    cos = [math.cos(2.0 * math.pi * r / n) for r in range(n)]

    total = 0.0
    c = 0.0  # Kahan compensation

    def kahan_add(value: float) -> None:
        nonlocal total, c
        y = value - c
        t = total + y
        c = (t - total) - y
        total = t

    # Enumerate compositions of d into n parts (counts of residues 0..n-1)
    # while carrying:
    #   sum_cos = sum_j cos(k_j)
    #   sum_mod = sum_j k_j (mod n)
    #   mult    = multinomial multiplicity (number of k-vectors with those counts)
    def rec(
        r: int,
        remaining: int,
        mult: int,
        sum_cos: float,
        sum_mod: int,
        any_nonzero: bool,
    ) -> None:
        if r == n - 1:
            cnt = remaining
            sum_cos2 = sum_cos + cnt * cos[r]
            sum_mod2 = (sum_mod + cnt * r) % n
            any2 = any_nonzero or (cnt > 0 and r != 0)
            if not any2:
                # This is the all-zero frequency vector; its eigenvalue is 1.
                return
            lam = (sum_cos2 + cos[sum_mod2]) / m
            term = mult / (1.0 - lam)
            kahan_add(term)
            return

        # Choose how many coordinates take value r.
        # Multiplicity update by choosing positions for this residue.
        row = _BINOM[remaining]
        cr = cos[r]
        for cnt in range(remaining + 1):
            rec(
                r + 1,
                remaining - cnt,
                mult * row[cnt],
                sum_cos + cnt * cr,
                (sum_mod + cnt * r) % n,
                any_nonzero or (cnt > 0 and r != 0),
            )

    rec(0, d, 1, 0.0, 0, False)
    return total


@lru_cache(maxsize=None)
def G(N: int, M: int) -> float:
    """G(N, M) = sum_{n=2..N} sum_{m=2..M} F(n, m)."""
    if N < 2 or M < 2:
        raise ValueError("N and M must be at least 2")

    total = 0.0
    c = 0.0

    def kahan_add(value: float) -> None:
        nonlocal total, c
        y = value - c
        t = total + y
        c = (t - total) - y
        total = t

    for n in range(2, N + 1):
        for m in range(2, M + 1):
            kahan_add(F(n, m))

    return total


def _self_test() -> None:
    # Test values given in the problem statement.
    eps = 1e-12

    assert abs(F(2, 2) - (1.0 / 2.0)) < eps
    assert abs(F(3, 2) - (4.0 / 3.0)) < eps
    assert abs(F(2, 3) - (9.0 / 4.0)) < eps
    assert abs(F(4, 5) - (6875.0 / 24.0)) < 1e-9

    assert abs(G(3, 3) - (137.0 / 12.0)) < eps
    assert abs(G(4, 5) - (6277.0 / 12.0)) < 1e-9

    # Given in scientific format with 12 digits after decimal.
    target = 1.681521567954e4
    assert abs(G(6, 6) - target) < 1e-6


def main() -> None:
    _self_test()
    ans = G(12, 12)
    print(_format_sci(ans))


if __name__ == "__main__":
    main()
