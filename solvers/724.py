#!/usr/bin/env python3
"""
Project Euler 724: Drone Delivery

Run:
  python3 main.py

The program prints the required answer (rounded to the nearest integer).

No external libraries are used (standard library only).
"""
from __future__ import annotations

import math
from fractions import Fraction


# Euler–Mascheroni constant (more digits than double precision needs)
_EULER_GAMMA = 0.5772156649015328606065120900824024310421


def _harmonic_exact(n: int) -> Fraction:
    """H_n = sum_{k=1..n} 1/k, exactly as a Fraction (for small n)."""
    s = Fraction(0, 1)
    for k in range(1, n + 1):
        s += Fraction(1, k)
    return s


def _harmonic2_exact(n: int) -> Fraction:
    """H_n^(2) = sum_{k=1..n} 1/k^2, exactly as a Fraction (for small n)."""
    s = Fraction(0, 1)
    for k in range(1, n + 1):
        s += Fraction(1, k * k)
    return s


def expected_distance_exact(n: int) -> Fraction:
    """
    Exact E(n) as a Fraction for small n.

    Derived closed form:
      E(n) = (n/2) * (H_n^2 + H_n^(2))
    """
    H = _harmonic_exact(n)
    H2 = _harmonic2_exact(n)
    return Fraction(n, 2) * (H * H + H2)


def _harmonic_asymptotic(n: int) -> float:
    """
    Asymptotic expansion for H_n with tiny error for large n.

    H_n = log(n) + gamma + 1/(2n) - 1/(12n^2) + 1/(120n^4) - 1/(252n^6) + ...
    """
    inv = 1.0 / n
    inv2 = inv * inv
    inv4 = inv2 * inv2
    inv6 = inv4 * inv2
    return (
        math.log(n)
        + _EULER_GAMMA
        + 0.5 * inv
        - (1.0 / 12.0) * inv2
        + (1.0 / 120.0) * inv4
        - (1.0 / 252.0) * inv6
    )


def _harmonic2_asymptotic(n: int) -> float:
    """
    Asymptotic expansion for H_n^(2) for large n.

    H_n^(2) = pi^2/6 - 1/n + 1/(2n^2) - 1/(6n^3) + 1/(30n^5) - 1/(42n^7) + ...
    """
    inv = 1.0 / n
    inv2 = inv * inv
    inv3 = inv2 * inv
    inv5 = inv3 * inv2
    inv7 = inv5 * inv2
    return (
        (math.pi * math.pi) / 6.0
        - inv
        + 0.5 * inv2
        - (1.0 / 6.0) * inv3
        + (1.0 / 30.0) * inv5
        - (1.0 / 42.0) * inv7
    )


def expected_distance(n: int) -> float:
    """
    Compute E(n) as a float.

    Uses the exact Fraction formula for small n (to support asserts),
    and asymptotic expansions for large n.
    """
    if n <= 2000:
        return float(expected_distance_exact(n))

    H = _harmonic_asymptotic(n)
    H2 = _harmonic2_asymptotic(n)
    return 0.5 * n * (H * H + H2)


def _round_nearest_int(x: float) -> int:
    """Round to nearest integer (x is positive here)."""
    return int(math.floor(x + 0.5))


def _self_test() -> None:
    # Test values from the problem statement
    assert expected_distance_exact(2) == Fraction(7, 2)
    assert expected_distance_exact(5) == Fraction(12019, 720)

    e100 = expected_distance(100)
    assert abs(e100 - 1427.193470) < 1e-6


def main() -> None:
    _self_test()

    n = 10**8
    e = expected_distance(n)
    print(_round_nearest_int(e))


if __name__ == "__main__":
    main()
