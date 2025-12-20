#!/usr/bin/env python3
"""
Project Euler 965: Expected Minimal Fractional Value

Let {x} be the fractional part of x.
For a fixed N, define f_N(x) = min({n x}) over integers n with 0 < n <= N.
Let F(N) be the expected value of f_N(x) for x uniformly sampled from [0, 1].

This program computes F(10^4) and prints it rounded to 13 digits after the decimal point.

Key idea:
The points where the minimizing n can change occur at adjacent fractions in the Farey sequence of order N.
If a/b < c/d are adjacent Farey fractions (in lowest terms), the contribution of interval [a/b, c/d) to F(N)
simplifies to 1 / (2 * b * d^2). Summing this over all adjacent pairs gives F(N).

We iterate through the Farey sequence in O(|F_N|) time using the standard "next term" recurrence, without sorting.
"""

from __future__ import annotations
import sys


def farey_integral_expected_min(N: int) -> float:
    """Return F(N) as a float, using a single pass over the Farey sequence of order N."""
    if N <= 0:
        raise ValueError("N must be positive")

    # Consecutive Farey fractions: a/b < c/d
    a, b = 0, 1
    c, d = 1, N

    # We sum many positive tiny terms; use chunked accumulation + compensated add for stability and speed.
    total = 0.0
    comp = 0.0
    partial = 0.0

    # Flush partial into total every 4096 terms (power of two for fast masking).
    mask = 4096 - 1
    cnt = 0

    while not (a == 1 and b == 1):
        # Contribution of interval [a/b, c/d) is 1/(2*b*d^2).
        # Use 0.5/(b*d*d) to reduce ops.
        partial += 0.5 / (b * d * d)

        cnt += 1
        if (cnt & mask) == 0:
            # Kahan-style compensated addition of the chunk.
            y = partial - comp
            t = total + y
            comp = (t - total) - y
            total = t
            partial = 0.0

        k = (N + b) // d

        # Update (a/b, c/d) -> (c/d, (k*c-a)/(k*d-b))
        a2 = c
        b2 = d
        c2 = k * c - a
        d2 = k * d - b
        a, b, c, d = a2, b2, c2, d2

    # Flush remaining partial
    if partial:
        y = partial - comp
        t = total + y
        comp = (t - total) - y
        total = t

    return total


def _self_test() -> None:
    # Test values given in the problem statement.
    v1 = farey_integral_expected_min(1)
    assert abs(v1 - 0.5) < 1e-15, v1

    v4 = farey_integral_expected_min(4)
    assert abs(v4 - 0.25) < 1e-15, v4

    v10 = farey_integral_expected_min(10)
    # Statement provides a decimal approximation.
    assert abs(v10 - 0.1319444444444) < 1e-12, v10


def main() -> None:
    _self_test()

    N = 10_000
    if len(sys.argv) >= 2:
        N = int(sys.argv[1])

    ans = farey_integral_expected_min(N)

    # Required output: rounded to 13 digits after the decimal point.
    print(f"{ans:.13f}")


if __name__ == "__main__":
    main()
