#!/usr/bin/env python3
"""
Project Euler 906 - A Collective Decision

We have 3 voters and n options. Each voter has a uniformly random ranking (a random permutation).
An option i is chosen iff it beats every other option by majority vote (i.e., is a Condorcet winner).
For 3 voters this is equivalent to: no other option is ranked above i by at least two voters.

This file computes P(20000) and prints it rounded to 10 digits after the decimal point.

No external libraries are used.
"""
from __future__ import annotations

import sys


def agreement_probability(n: int, *, eps_row: float = 1e-20) -> float:
    """
    Return P(n): probability that a Condorcet winner exists for 3 random permutations of {1..n}.

    Method (sketch):
      - Fix a candidate i. Let a,b,c be the number of candidates ranked above i by voters 1,2,3.
      - i is a Condorcet winner iff these three "above sets" are pairwise disjoint.
      - With N=n-1 remaining candidates:
            Pr(disjoint | a,b,c) = C(N-a, b)/C(N,b) * C(N-a-b, c)/C(N,c)
        (0 if the binomials are invalid).
      - Summing over c can be done in closed form:
            sum_{c=0}^{N-(a+b)} C(N-(a+b), c)/C(N,c) = (N+1)/(a+b+1)
      - This reduces P(n) to a positive double sum over (a,b).
      - The inner ratio can be updated multiplicatively without factorials.

    eps_row:
      - We stop the inner loop early when the remaining tail of the current row is provably
        < eps_row (using monotonicity), which is more than enough for 10-decimal output.
    """
    if n < 1:
        raise ValueError("n must be a positive integer")
    if n == 1:
        return 1.0  # trivially "agree" on the only option

    N = n - 1
    nn = float(n * n)
    factor = float(N + 1) / nn

    # Kahan summation for improved numerical stability.
    s = 0.0
    c = 0.0

    for a in range(N + 1):
        bmax = N - a

        # b = 0 term: g = 1
        term = 1.0 / (a + 1)
        y = term - c
        t = s + y
        c = (t - s) - y
        s = t

        # g(a,b) = C(N-a,b)/C(N,b) updated by:
        # g_{b} = g_{b-1} * (N-a-(b-1)) / (N-(b-1))
        g = 1.0
        num = N - a  # starts at (N-a-(b-1)) for b=1
        den = N  # starts at (N-(b-1))   for b=1

        for b in range(1, bmax + 1):
            g *= num / den
            num -= 1
            den -= 1

            term = g / (a + b + 1)

            y = term - c
            t = s + y
            c = (t - s) - y
            s = t

            rem = bmax - b
            # For fixed a, g(a,b) decreases with b (probability a random b-subset avoids a fixed a-subset),
            # and (a+b+1) increases, so term is decreasing as well.
            # Hence the remaining tail is < term * rem.
            if rem and term * rem < eps_row:
                break

    return factor * s


def _self_test() -> None:
    # Test values from the problem statement:
    p3 = agreement_probability(3)
    assert abs(p3 - (17.0 / 18.0)) < 1e-15

    p10 = agreement_probability(10)
    # Given as an approximation in the statement.
    assert abs(p10 - 0.6760292265) < 2e-11


def main(argv: list[str]) -> None:
    _self_test()

    n = 20000
    if len(argv) >= 2:
        n = int(argv[1])

    p = agreement_probability(n)
    print(f"{p:.10f}")


if __name__ == "__main__":
    main(sys.argv)
