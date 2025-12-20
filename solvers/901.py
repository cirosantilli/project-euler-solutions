#!/usr/bin/env python3
"""Project Euler 901: Well Drilling

A driller chooses depths d1, d2, ... (positive reals). Each attempt drills from
ground level to the chosen depth and costs exactly that many hours.

The groundwater depth X is unknown but fixed across attempts and is distributed
as Exp(1): P(X > d) = e^{-d}.

If an attempt drills to depth d and finds no water, then we learn X > d and the
next attempt must use a deeper depth (otherwise it would surely fail).

This script computes the minimal expected drilling time under an optimal
strategy and prints the result rounded to 9 decimal places.

No external libraries are used.
"""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_UP, getcontext


def expected_time_stationary_float(d1: float) -> float:
    """Evaluate the expected time for the stationary optimality-recurrence.

    For an optimal sequence d0=0 < d1 < d2 < ..., the first-order conditions for
    minimizing E[T] imply the recurrence

        d_{n+1} = exp(d_n - d_{n-1})  for n >= 1.

    Under this recurrence, the expected time simplifies to:

        E[T] = d1 + 1 + sum_{n>=1} exp(-d_n)

    (the n=2 term equals 1 because d2 = exp(d1)).

    If the generated sequence ever fails to be strictly increasing, this d1 is
    invalid for a feasible strategy and we return +inf.

    This float version is used only to bracket the minimum quickly.
    """

    if d1 <= 0.0:
        return float("inf")

    d_prev = 0.0
    d = d1
    total = d1 + 1.0

    # The sum converges extremely fast; once exp(-d) is tiny, remaining terms
    # are negligible for our purposes.
    eps = 1e-16

    for _ in range(10000):
        term = math.exp(-d)
        total += term
        if term < eps:
            break

        inc = d - d_prev
        # If the increment is large, the next depth is astronomical and the
        # corresponding exp(-depth) terms are far below eps.
        if inc > 80.0:
            break

        d_next = math.exp(inc)
        if d_next <= d:
            return float("inf")

        d_prev, d = d, d_next

    return total


def expected_time_stationary_decimal(d1: Decimal, eps: Decimal) -> Decimal:
    """High-precision version of expected_time_stationary_float."""

    if d1 <= 0:
        return Decimal("Infinity")

    d_prev = Decimal(0)
    d = d1
    total = d1 + Decimal(1)

    for _ in range(20000):
        term = (-d).exp()
        total += term
        if term < eps:
            break

        inc = d - d_prev
        if inc > Decimal(80):
            break

        d_next = inc.exp()
        if d_next <= d:
            return Decimal("Infinity")

        d_prev, d = d, d_next

    return total


def golden_section_min(
    a: Decimal, b: Decimal, iters: int, eps: Decimal
) -> tuple[Decimal, Decimal]:
    """Golden-section search for the minimum of a unimodal function on [a, b]."""

    sqrt5 = Decimal(5).sqrt()
    phi = (Decimal(1) + sqrt5) / Decimal(2)
    invphi = Decimal(1) / phi

    c = b - (b - a) * invphi
    d = a + (b - a) * invphi

    fc = expected_time_stationary_decimal(c, eps)
    fd = expected_time_stationary_decimal(d, eps)

    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - (b - a) * invphi
            fc = expected_time_stationary_decimal(c, eps)
        else:
            a, c, fc = c, d, fd
            d = a + (b - a) * invphi
            fd = expected_time_stationary_decimal(d, eps)

    x = (a + b) / 2
    fx = expected_time_stationary_decimal(x, eps)
    return x, fx


def find_bracket_by_coarse_scan() -> tuple[float, float, float]:
    """Coarsely scan d1 to locate a good bracket around the global minimum."""

    best_x = 0.1
    best_f = float("inf")

    # Searching up to 3 is plenty: for d1 > 3, E[T] >= d1 + 1 > 4.
    lo, hi = 1e-4, 3.0
    step = 5e-4

    x = lo
    while x <= hi + 1e-15:
        fx = expected_time_stationary_float(x)
        if fx < best_f:
            best_f = fx
            best_x = x
        x += step

    # Build a symmetric bracket around the best point, but keep it inside a
    # region where the recurrence stays strictly increasing (otherwise the
    # objective is treated as +inf). A small bracket around the best sample is
    # typically unimodal, making golden-section search reliable.
    def finite(t: float) -> bool:
        return math.isfinite(expected_time_stationary_float(t))

    width = 0.05
    while True:
        a = max(lo, best_x - width)
        b = min(hi, best_x + width)
        m = (a + b) / 2
        q1 = a + (b - a) * 0.25
        q3 = a + (b - a) * 0.75

        if finite(a) and finite(q1) and finite(m) and finite(q3) and finite(b):
            fm = expected_time_stationary_float(m)
            if fm <= expected_time_stationary_float(
                a
            ) and fm <= expected_time_stationary_float(b):
                return a, b, best_x

        width *= 0.6
        if width < 0.002:
            return max(lo, best_x - 0.005), min(hi, best_x + 0.005), best_x


def solve() -> Decimal:
    # Use enough precision that rounding to 9 decimals is safe.
    getcontext().prec = 120

    a_f, b_f, _ = find_bracket_by_coarse_scan()

    a = Decimal(str(a_f))
    b = Decimal(str(b_f))

    eps = Decimal("1e-90")

    _, f_opt = golden_section_min(a, b, iters=250, eps=eps)
    return f_opt


def main() -> None:
    ans = solve().quantize(Decimal("1e-9"), rounding=ROUND_HALF_UP)
    print(ans)


if __name__ == "__main__":
    main()
