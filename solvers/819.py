#!/usr/bin/env python3
"""
Project Euler 819: Iterative Sampling

We start with the n-tuple (1,2,...,n). Each step creates a new n-tuple by
independently sampling each position from the multiset of values in the
previous tuple (i.e., sampling from the empirical distribution).

Let E(n) be the expected number of steps until all entries are equal.

This implementation uses only the Python standard library.
"""

from __future__ import annotations


def expected_steps(n: int) -> float:
    """
    Compute E(n) as a float.

    Key reduction:
      Starting from all distinct labels, the process can be viewed as repeated
      composition of random functions f:[n]->[n]. The only state we need is
      m = size of the image of the composed function (equivalently, the number
      of distinct "ancestral labels" still present). Given m, the next image
      size is the number of occupied bins when throwing m balls uniformly into
      n bins.

    The chain is monotone non-increasing in m, so the expected hitting time to
    m=1 is computed by a triangular dynamic program.
    """
    if n <= 1:
        return 0.0

    # T[m] = expected remaining steps to reach 1 distinct label, starting from m
    T = [0.0] * (n + 1)
    T[1] = 0.0

    # p[k] = P( number of occupied bins = k ) for the current m (updated in place)
    # Start with m=0: p[0]=1.
    p = [0.0] * (n + 1)
    p[0] = 1.0
    inv_n = 1.0 / n

    for m in range(1, n + 1):
        # Update occupancy distribution from m-1 balls to m balls:
        # new_p[k] = old_p[k] * (k/n) + old_p[k-1] * ((n-k+1)/n)
        # Do it in-place descending so old_p[k-1] is still available.
        for k in range(m, 0, -1):
            p[k] = p[k] * (k * inv_n) + p[k - 1] * ((n - k + 1) * inv_n)
        p[0] = 0.0

        if m == 1:
            continue

        stay = p[m]  # probability of no collisions -> state stays at m
        acc = 0.0
        for k in range(1, m):
            acc += p[k] * T[k]

        # T[m] = 1 + stay*T[m] + sum_{k<m} p[k]*T[k]
        T[m] = (1.0 + acc) / (1.0 - stay)

    return T[n]


def expected_steps_exact(n: int):
    """
    Exact E(n) using Fractions (intended only for small n in self-tests).
    """
    from fractions import Fraction

    if n <= 1:
        return Fraction(0)

    T = [Fraction(0) for _ in range(n + 1)]
    T[1] = Fraction(0)

    # p[k] for current m, as exact fractions
    p = [Fraction(0) for _ in range(n + 1)]
    p[0] = Fraction(1)

    for m in range(1, n + 1):
        new = [Fraction(0) for _ in range(n + 1)]
        for k in range(1, m + 1):
            new[k] = p[k] * Fraction(k, n) + p[k - 1] * Fraction(n - k + 1, n)
        p = new

        if m == 1:
            continue

        stay = p[m]
        numer = Fraction(1)
        for k in range(1, m):
            numer += p[k] * T[k]
        T[m] = numer / (1 - stay)

    return T[n]


def _self_test() -> None:
    """
    Asserts the check values from the problem statement.
    """
    from fractions import Fraction

    assert expected_steps_exact(3) == Fraction(27, 7)
    assert expected_steps_exact(5) == Fraction(468125, 60701)
    assert round(float(expected_steps_exact(5)), 6) == 7.711982

    # Optional cross-check: float path matches exact for these small n
    assert abs(expected_steps(3) - float(Fraction(27, 7))) < 1e-12
    assert abs(expected_steps(5) - float(Fraction(468125, 60701))) < 1e-12


def main() -> None:
    _self_test()
    n = 10**3
    ans = expected_steps(n)
    print(f"{ans:.6f}")


if __name__ == "__main__":
    main()
