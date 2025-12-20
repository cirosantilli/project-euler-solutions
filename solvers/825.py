#!/usr/bin/env python3
"""Project Euler 825: Chasing Game

Computes T(10^14) and prints it rounded to 8 digits after the decimal point.

No third-party libraries are used.

The program includes asserts for the example/test values given in the problem
statement.

Usage:
  python3 main.py
  python3 main.py <N>
"""

from __future__ import annotations

import math
import sys
from fractions import Fraction
from typing import List


# ---------------------------------------------------------------------------
# Exact solver for small n (Fractions) for the statement's test values
# ---------------------------------------------------------------------------


def _gauss_solve_fraction(aug: List[List[Fraction]]) -> List[Fraction]:
    """Solve a linear system with an augmented matrix [A|b] using Fractions."""
    n = len(aug)
    m = len(aug[0])  # n + 1
    r = 0
    for c in range(n):
        piv = None
        for i in range(r, n):
            if aug[i][c] != 0:
                piv = i
                break
        if piv is None:
            continue
        if piv != r:
            aug[r], aug[piv] = aug[piv], aug[r]

        pv = aug[r][c]
        for j in range(c, m):
            aug[r][j] /= pv

        for i in range(n):
            if i == r:
                continue
            f = aug[i][c]
            if f == 0:
                continue
            for j in range(c, m):
                aug[i][j] -= f * aug[r][j]

        r += 1
        if r == n:
            break

    return [aug[i][-1] for i in range(n)]


def S_fraction(n: int) -> Fraction:
    """Exact S(n) for small n, by solving the Markov equations with Fractions."""
    if n < 2:
        raise ValueError("n must be >= 2")

    L = 2 * n
    m = L - 1  # states d = 1..L-1

    # Let f[d] be the probability that the current mover eventually wins
    # when the opponent is d steps ahead (1 <= d <= L-1).
    # Then S(n) = 2*f[n] - 1.

    A = [[Fraction(0) for _ in range(m + 1)] for __ in range(m)]

    def idx(d: int) -> int:
        return d - 1

    for d in range(1, m + 1):
        row = A[idx(d)]
        if d == 1:
            # always catch immediately
            row[idx(1)] = Fraction(1)
            row[-1] = Fraction(1)
        elif d == 2:
            # f2 = 2/3 + (1/3)*(1 - f_{L-1})
            # -> f2 + (1/3) f_{L-1} = 1
            row[idx(2)] = Fraction(1)
            row[idx(L - 1)] += Fraction(1, 3)
            row[-1] = Fraction(1)
        elif d == 3:
            # f3 = 1/3 + (1/3)*(1 - f_{L-2}) + (1/3)*(1 - f_{L-1})
            # -> f3 + (1/3) f_{L-2} + (1/3) f_{L-1} = 1
            row[idx(3)] = Fraction(1)
            row[idx(L - 2)] += Fraction(1, 3)
            row[idx(L - 1)] += Fraction(1, 3)
            row[-1] = Fraction(1)
        else:
            # If d >= 4, current mover cannot catch immediately.
            # After rolling k in {1,2,3}, the opponent's turn begins with
            # opponent behind by (L - (d-k)) = L - d + k.
            # Current mover wins with probability 1 - f[L - d + k].
            # -> f[d] = 1 - (f[L-d+1] + f[L-d+2] + f[L-d+3]) / 3
            row[idx(d)] = Fraction(1)
            for k in (1, 2, 3):
                row[idx(L - d + k)] += Fraction(1, 3)
            row[-1] = Fraction(1)

    sol = _gauss_solve_fraction(A)
    return 2 * sol[idx(n)] - 1


# ---------------------------------------------------------------------------
# Fast S(n) in O(1) time (floating point), using the closed form.
# Used only for n up to ~50 when computing a convergent correction constant.
# ---------------------------------------------------------------------------


def _gauss_solve_float(mat: List[List[float]], rhs: List[float]) -> List[float]:
    """Solve a 4x4 linear system with partial pivoting."""
    n = 4
    aug = [mat[i][:] + [rhs[i]] for i in range(n)]

    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(aug[r][c]))
        if abs(aug[piv][c]) < 1e-300:
            raise ValueError("Singular 4x4 system")
        if piv != c:
            aug[c], aug[piv] = aug[piv], aug[c]

        pv = aug[c][c]
        for j in range(c, n + 1):
            aug[c][j] /= pv

        for r in range(n):
            if r == c:
                continue
            f = aug[r][c]
            if f == 0.0:
                continue
            for j in range(c, n + 1):
                aug[r][j] -= f * aug[c][j]

    return [aug[i][-1] for i in range(n)]


def S_fast_float(n: int) -> float:
    """Compute S(n) as a float via a closed form + 4 boundary equations."""
    if n < 2:
        raise ValueError("n must be >= 2")
    if n == 2:
        return float(S_fraction(2))

    L = 2 * n
    q = -2.0 + math.sqrt(3.0)  # |q| < 1

    def g(y: int, A: float, B: float, C: float, E: float) -> float:
        return A + B * y + C * (q**y) + E * (q ** (L - y))

    def row(y: int) -> List[float]:
        return [
            1.0,
            float(y),
            q**y,
            q ** (L - y),
        ]

    # Build the 4 linear equations in unknowns [A, B, C, E].
    #
    # g(2) + (1/3) g(L-1) = 1
    # g(3) + (1/3) g(L-2) + (1/3) g(L-1) = 1
    # g(L-1) + (1/3)g(2) + (1/3)g(3) + (1/3)g(4) = 1
    # g(L-2) + (1/3)g(3) + (1/3)g(4) + (1/3)g(5) = 1

    def add_scaled(dst: List[float], scale: float, src: List[float]) -> None:
        for i in range(4):
            dst[i] += scale * src[i]

    mat: List[List[float]] = []
    rhs: List[float] = []

    # Eq1
    r1 = [0.0, 0.0, 0.0, 0.0]
    add_scaled(r1, 1.0, row(2))
    add_scaled(r1, 1.0 / 3.0, row(L - 1))
    mat.append(r1)
    rhs.append(1.0)

    # Eq2
    r2 = [0.0, 0.0, 0.0, 0.0]
    add_scaled(r2, 1.0, row(3))
    add_scaled(r2, 1.0 / 3.0, row(L - 2))
    add_scaled(r2, 1.0 / 3.0, row(L - 1))
    mat.append(r2)
    rhs.append(1.0)

    # Eq3
    r3 = [0.0, 0.0, 0.0, 0.0]
    add_scaled(r3, 1.0, row(L - 1))
    add_scaled(r3, 1.0 / 3.0, row(2))
    add_scaled(r3, 1.0 / 3.0, row(3))
    add_scaled(r3, 1.0 / 3.0, row(4))
    mat.append(r3)
    rhs.append(1.0)

    # Eq4
    r4 = [0.0, 0.0, 0.0, 0.0]
    add_scaled(r4, 1.0, row(L - 2))
    add_scaled(r4, 1.0 / 3.0, row(3))
    add_scaled(r4, 1.0 / 3.0, row(4))
    add_scaled(r4, 1.0 / 3.0, row(5))
    mat.append(r4)
    rhs.append(1.0)

    A, B, C, E = _gauss_solve_float(mat, rhs)

    gn = g(n, A, B, C, E)
    return 2.0 * gn - 1.0


# ---------------------------------------------------------------------------
# Digamma (psi) and shifted harmonic sum
# ---------------------------------------------------------------------------


def digamma(x: float) -> float:
    """Compute psi(x) for x>0 via recurrence + asymptotic Bernoulli expansion."""
    if x <= 0.0:
        raise ValueError("digamma(x) requires x > 0")

    res = 0.0
    while x < 8.0:
        res -= 1.0 / x
        x += 1.0

    inv = 1.0 / x
    inv2 = inv * inv

    # Asymptotic series (Bernoulli numbers up to B12):
    res += math.log(x) - 0.5 * inv

    t = inv2
    res -= t / 12.0
    t *= inv2
    res += t / 120.0
    t *= inv2
    res -= t / 252.0
    t *= inv2
    res += t / 240.0
    t *= inv2
    res -= t / 132.0
    t *= inv2
    res += t * (691.0 / 32760.0)

    return res


def shifted_harmonic_sum(N: int, c: float) -> float:
    """Sum_{k=1..N-1} 1/(k+c) = psi(N+c) - psi(1+c)."""
    return digamma(float(N) + c) - digamma(1.0 + c)


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------


def correction_constant(max_n: int = 60) -> float:
    """K = Σ_{n=2..∞} (S(n) - 1/(n-1+c)), approximated by a short finite sum."""
    c = (3.0 - math.sqrt(3.0)) / 6.0
    acc = 0.0
    for n in range(2, max_n + 1):
        acc += S_fast_float(n) - 1.0 / (n - 1 + c)
    return acc


def T(N: int) -> float:
    c = (3.0 - math.sqrt(3.0)) / 6.0
    K = correction_constant()
    return shifted_harmonic_sum(N, c) + K


def _self_test() -> None:
    # Test values from the problem statement.
    assert S_fraction(2) == Fraction(7, 11)

    # T(10) = 2.38235282 when rounded to 8 digits after the decimal point.
    t10 = sum(S_fraction(n) for n in range(2, 11))
    lo = Fraction(2382352815, 10**9)  # 2.382352815
    hi = Fraction(2382352825, 10**9)  # 2.382352825
    assert lo <= t10 < hi


def main() -> None:
    _self_test()
    N = 10**14
    if len(sys.argv) > 1:
        N = int(sys.argv[1])
    ans = T(N)
    print(f"{ans:.8f}")


if __name__ == "__main__":
    main()
