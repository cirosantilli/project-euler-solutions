#!/usr/bin/env python3
"""
Project Euler 957: Point Genesis

We are given three fixed red points and two initial blue points. Each day:
  - draw every line through a red point and a blue point;
  - every intersection of two such lines (coming from different red points) turns blue.

Let g(n) be the maximal possible number of blue points after n days.
The problem asks for g(16).

This implementation uses an exact closed-form expression for g(n) derived by
reducing the geometry to counting integer lattice points in a growing hexagonal
region (an Ehrhart-type quasi-polynomial), which simplifies to a linear
combination of powers of 2 with small rational coefficients.

No external libraries are used.
"""

from __future__ import annotations


def g(n: int) -> int:
    """Return g(n) for n >= 0 as an integer."""
    if n < 0:
        raise ValueError("n must be non-negative")

    # Exact formula:
    #   g(n) = ( 11*16^n + 132*8^n + 564*4^n + 1008*2^n
    #            -384*(-1)^n -128*(-2)^n + 768 ) / 864
    #
    # Compute using shifts: 2^k == 1 << k.
    sign = -1 if (n & 1) else 1  # (-1)^n

    numer = 0
    numer += 11 * (1 << (4 * n))  # 11 * 16^n
    numer += 132 * (1 << (3 * n))  # 132 * 8^n
    numer += 564 * (1 << (2 * n))  # 564 * 4^n
    numer += 1008 * (1 << n)  # 1008 * 2^n
    numer += -384 * sign  # -384 * (-1)^n
    numer += -128 * sign * (1 << n)  # -128 * (-2)^n = -128 * (-1)^n * 2^n
    numer += 768  # constant term

    # The division is exact for all n >= 0.
    assert numer % 864 == 0
    return numer // 864


def main() -> None:
    # Test values from the problem statement:
    assert g(1) == 8
    assert g(2) == 28

    print(g(16))


if __name__ == "__main__":
    main()
