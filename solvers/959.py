#!/usr/bin/env python3
"""Project Euler 959: Asymmetric Random Walk

A frog starts at 0 on the number line. Each step is either:
  - jump a units left  (delta = -a)
  - jump b units right (delta = +b)
with probability 1/2 each.

Let c_n be the expected number of distinct positions visited in the first n steps.
Define
  f(a, b) = lim_{n->infinity} c_n / n.

This program computes f(89, 97) and prints it rounded to nine digits after the
decimal point.

No third-party libraries are used.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP, getcontext
from math import gcd


def _round_9(x: Decimal) -> str:
    """Round to exactly nine digits after the decimal point (half-up)."""
    return str(x.quantize(Decimal("0.000000001"), rounding=ROUND_HALF_UP))


def f(a: int, b: int, *, prec: int = 120, term_cutoff_exp: int = 70) -> Decimal:
    """Compute f(a,b) as a Decimal.

    We reduce by gcd(a,b) because scaling the lattice does not change the count
    of distinct visited positions.

    For a != b the walk has nonzero drift, is transient, and
      f(a,b) = 1 / G
    where G is the expected number of visits to the origin.

    For this 2-step walk,
      G = sum_{k>=0} C((a+b)k, ak) / 2^{(a+b)k}.

    The series is summed via a term-to-term recurrence.
    """
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be positive integers")

    g = gcd(a, b)
    a //= g
    b //= g

    if a == b:
        # Zero drift => recurrent in 1D => c_n grows sublinearly => limit 0.
        return Decimal(0)

    # Symmetry: mirroring the walk swaps (a,b) without changing unique counts.
    if a > b:
        a, b = b, a

    m = a + b

    # Use a high precision context for safe rounding to 9 decimal places.
    getcontext().prec = prec

    # S = sum_{k>=0} t_k, t_0 = 1, t_k = C(mk, ak) / 2^{mk}.
    # Recurrence:
    #   t_{k+1} = t_k * 2^{-m} * prod_{i=1..m} (mk+i)
    #                    / ( prod_{i=1..a} (ak+i) * prod_{i=1..b} (bk+i) )
    inv_pow2m = Decimal(2) ** (-m)  # exact terminating decimal

    t = Decimal(1)
    S = t

    eps = Decimal(10) ** (-term_cutoff_exp)

    k = 0
    while True:
        mk = m * k
        ak = a * k
        bk = b * k

        ratio = inv_pow2m
        i_num = i_a = i_b = 1

        # Interleave multiplications/divisions to keep intermediate magnitudes tame.
        while i_num <= m or i_a <= a or i_b <= b:
            if i_num <= m:
                ratio *= mk + i_num
                i_num += 1
            if i_a <= a:
                ratio /= ak + i_a
                i_a += 1
            if i_b <= b:
                ratio /= bk + i_b
                i_b += 1

        t *= ratio
        S += t
        k += 1

        if t < eps:
            break
        if k > 20000:
            raise RuntimeError("Series did not converge fast enough")

    return Decimal(1) / S


def _self_test() -> None:
    # Values given in the problem statement.
    assert f(1, 1) == Decimal(0)
    assert _round_9(f(1, 2)) == "0.427050983"


def main() -> None:
    _self_test()
    ans = f(89, 97)
    print(_round_9(ans))


if __name__ == "__main__":
    main()
