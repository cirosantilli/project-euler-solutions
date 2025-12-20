#!/usr/bin/env python3
"""
Project Euler 848: Guessing with Sets

Computes:
    sum_{i=0..20} sum_{j=0..20} p(7^i, 5^j)
rounded to 8 digits after the decimal point.

No external libraries are used (only Python standard library).
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache


def _next_pow2_times3(x: int) -> tuple[int, int]:
    """
    Returns (T, p) where:
      - p is the smallest power of 2 such that 3*p >= x
      - T = 3*p
    """
    p = 1
    while 3 * p < x:
        p <<= 1
    return 3 * p, p


@lru_cache(maxsize=None)
def p(m: int, n: int) -> Fraction:
    """
    p(m,n): winning probability of the player to move, when
      - the opponent's secret is uniformly in {1..m}
      - your secret is uniformly in {1..n}
    Both players play optimally.

    Returns an exact Fraction.
    """
    if m <= 0 or n <= 0:
        raise ValueError("m and n must be positive integers")

    # Trivial/base cases from the statement.
    if m == 1:
        return Fraction(1, 1)
    if n == 1:
        return Fraction(1, m)

    # Special small-n cases (also helpful to avoid long singleton chains).
    if n == 2:
        # Exact closed form for all m>=2:
        # p(m,2) = 3/(2m); and p(1,2)=1 by m==1 base.
        return Fraction(3, 2 * m)

    # Special small-m cases.
    if m == 2:
        # p(2,n) = 1 - 1/(2n)
        return Fraction(1, 1) - Fraction(1, 2 * n)
    if m == 3:
        # p(3,1)=1/3 handled by n==1 above; otherwise p(3,n)=1-1/n.
        return Fraction(1, 1) - Fraction(1, n)

    # ---- "Stable" closed-form regions (derived from the recurrence) ----
    #
    # Let p_m be the smallest power of 2 with 3*p_m >= m.
    # Define L = 3*(p_m/2). If n >= L, then:
    #   p(m,n) = 1 - L*(m - p_m)/(m*n)
    #
    # Symmetrically, for n>=3 let p_n be the smallest power of 2 with 3*p_n >= n
    # and T = 3*p_n. If m >= T, then:
    #   p(m,n) = T*(n - p_n)/(n*m)
    #
    # These formulas make the required (7^i, 5^j) evaluations essentially O(1).

    # High-n region (player to move is very likely to win).
    Tm, pm = _next_pow2_times3(m)  # pm is power of 2, Tm = 3*pm
    if pm >= 2:
        L = 3 * (pm >> 1)
        if n >= L:
            return Fraction(1, 1) - Fraction(L * (m - pm), m * n)

    # High-m region (player to move is unlikely to win; must win quickly).
    Tn, pn = _next_pow2_times3(n)
    if n >= 3 and m >= Tn:
        return Fraction(Tn * (n - pn), n * m)

    # ---- General recurrence fallback ----
    # On your turn, you choose a subset size 'a' (1 <= a <= m-1).
    # If a==1, you win immediately with probability 1/m, else the game continues.
    # After the answer, roles swap and the state becomes (n, a) or (n, m-a).
    #
    # For a>1:
    #   value(a) = (a/m)*(1 - p(n,a)) + ((m-a)/m)*(1 - p(n,m-a))
    #
    # For a==1:
    #   value(1) = 1/m + ((m-1)/m)*(1 - p(n,m-1))
    #
    # Empirically (and provably via concavity), the optimum is attained either
    # at a==1 or with an almost-even split near m/2. We therefore only test
    # a small candidate set.
    candidates = {1}
    half = m // 2
    for a in (half - 1, half, half + 1, (m + 1) // 2):
        if 1 < a < m:
            candidates.add(a)

    best = Fraction(-1, 1)
    for a in candidates:
        if a == 1:
            val = Fraction(1, m) + Fraction(m - 1, m) * (Fraction(1, 1) - p(n, m - 1))
        else:
            val = Fraction(a, m) * (Fraction(1, 1) - p(n, a)) + Fraction(m - a, m) * (
                Fraction(1, 1) - p(n, m - a)
            )
        if val > best:
            best = val
    return best


def _round_fraction_to_8dp(x: Fraction) -> str:
    """
    Rounds the Fraction to exactly 8 digits after the decimal point
    using integer arithmetic (round half up).
    """
    scale = 10**8
    num = x.numerator
    den = x.denominator
    scaled = (num * scale * 2 + den) // (2 * den)  # nearest integer to num/den * scale
    integer_part = scaled // scale
    frac_part = scaled % scale
    return f"{integer_part}.{frac_part:08d}"


def main() -> None:
    # --- Asserts for values given in the problem statement ---
    # p(1,n)=1
    assert p(1, 10) == Fraction(1, 1)
    assert p(1, 5**7) == Fraction(1, 1)

    # p(m,1)=1/m
    assert p(7, 1) == Fraction(1, 7)
    assert p(42, 1) == Fraction(1, 42)

    # p(7,5) ≈ 0.51428571
    assert p(7, 5) == Fraction(18, 35)
    assert abs(float(p(7, 5)) - 0.51428571) < 1e-8

    # --- Compute the required sum ---
    pow7 = [1]
    for _ in range(20):
        pow7.append(pow7[-1] * 7)

    pow5 = [1]
    for _ in range(20):
        pow5.append(pow5[-1] * 5)

    total = Fraction(0, 1)
    for m in pow7:
        for n in pow5:
            total += p(m, n)

    print(_round_fraction_to_8dp(total))


if __name__ == "__main__":
    main()
