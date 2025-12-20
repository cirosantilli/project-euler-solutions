#!/usr/bin/env python3
"""
Project Euler 807 - Loops of Ropes

Compute P(n): the probability that the red loop and blue loop can be separated.
The key reduction is:
    P(n) = A(2n-1, n) / (2n-1)!
where A(m,k) is an Eulerian number. This is OEIS A025585(n) divided by (2n-1)!.

We compute the central Eulerian number using the classic alternating-sum formula,
which is exact with Python big integers, then format the result rounded to
10 digits after the decimal point.
"""

from __future__ import annotations

import math
from fractions import Fraction


def central_eulerian(n: int) -> int:
    """
    Return a(n) = A(2n-1, n), also known as OEIS A025585(n).

    Closed form (integer):
        a(n) = sum_{j=0..n} (-1)^j * C(2n, j) * (n-j)^(2n-1)

    Note: the j=n term is 0^(2n-1)=0, so the loop may stop at n-1.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    m = 2 * n - 1
    total = 0
    for j in range(0, n):  # j=n contributes 0
        base = n - j
        term = math.comb(2 * n, j) * pow(base, m)
        if j & 1:
            total -= term
        else:
            total += term
    return total


def p_fraction(n: int) -> Fraction:
    """Exact rational P(n) as a Fraction."""
    num = central_eulerian(n)
    den = math.factorial(2 * n - 1)
    return Fraction(num, den)


def p_rounded_str(n: int, digits: int = 10) -> str:
    """
    Return P(n) rounded to `digits` digits after the decimal point, as a string.

    Uses exact integer arithmetic for correct rounding (half-up).
    """
    if digits < 0:
        raise ValueError("digits must be non-negative")
    num = central_eulerian(n)
    den = math.factorial(2 * n - 1)

    scale = 10**digits
    scaled = num * scale
    q, r = divmod(scaled, den)

    # Round half-up: if remainder >= 1/2, bump.
    if 2 * r >= den:
        q += 1

    if digits == 0:
        return str(q)

    # P(n) is in (0,1), but keep this generic.
    int_part = q // scale
    frac_part = q % scale
    return f"{int_part}.{frac_part:0{digits}d}"


def _self_test() -> None:
    # Test values given in the problem statement:
    assert p_fraction(3) == Fraction(11, 20)
    assert p_rounded_str(5, 10) == "0.4304177690"


def main() -> None:
    _self_test()
    print(p_rounded_str(80, 10))


if __name__ == "__main__":
    main()
