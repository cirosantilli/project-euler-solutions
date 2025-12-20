#!/usr/bin/env python3
"""
Project Euler 737: Coin Loops

No external libraries are used.

The program prints the number of coins needed to loop a given number of times.
Default: 2020 loops.

It also contains asserts for the check values stated in the problem.
"""

from __future__ import annotations

import math
import sys
from array import array


def _compute_euler_gamma(n: int = 2_000_000) -> float:
    """
    Compute Euler-Mascheroni constant gamma using:
      H_n = log(n) + gamma + 1/(2n) - 1/(12n^2) + 1/(120n^4) - 1/(252n^6) + 1/(240n^8) + ...
    => gamma ~= H_n - log(n) - 1/(2n) + 1/(12n^2) - 1/(120n^4) + 1/(252n^6) - 1/(240n^8)

    Using n=2,000,000 is fast and yields more than enough precision for double floats.
    """
    H = 0.0
    for k in range(1, n + 1):
        H += 1.0 / k

    inv = 1.0 / n
    inv2 = inv * inv
    inv4 = inv2 * inv2
    inv6 = inv4 * inv2
    inv8 = inv4 * inv4
    return (
        H
        - math.log(n)
        - 0.5 * inv
        + (1.0 / 12.0) * inv2
        - (1.0 / 120.0) * inv4
        + (1.0 / 252.0) * inv6
        - (1.0 / 240.0) * inv8
    )


_EULER_GAMMA = _compute_euler_gamma()


def _harmonic_asymp(x: float) -> float:
    """
    Euler-Maclaurin asymptotic expansion for H_x (for large x).
    """
    inv = 1.0 / x
    inv2 = inv * inv
    inv4 = inv2 * inv2
    inv6 = inv4 * inv2
    inv8 = inv4 * inv4
    # log(x) + gamma + 1/(2x) - 1/(12x^2) + 1/(120x^4) - 1/(252x^6) + 1/(240x^8)
    return (
        math.log(x)
        + _EULER_GAMMA
        + 0.5 * inv
        - (1.0 / 12.0) * inv2
        + (1.0 / 120.0) * inv4
        - (1.0 / 252.0) * inv6
        + (1.0 / 240.0) * inv8
    )


def _gauss_legendre(n: int) -> tuple[list[float], list[float]]:
    """
    Compute Gauss-Legendre nodes and weights on [-1, 1] for order n.

    Standard Newton iteration on Legendre polynomials:
      initial guess x_i ~= cos(pi*(i-0.25)/(n+0.5))
      refine with x <- x - P_n(x)/P'_n(x)

    Returns (nodes, weights) as Python lists, sorted ascending by node.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    x = [0.0] * n
    w = [0.0] * n

    m = (n + 1) // 2  # roots are symmetric
    for i in range(1, m + 1):
        # Initial guess
        z = math.cos(math.pi * (i - 0.25) / (n + 0.5))

        # Newton iterations
        for _ in range(50):
            p1 = 1.0
            p2 = 0.0
            for j in range(1, n + 1):
                p3 = p2
                p2 = p1
                p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j
            # p1 = P_n(z), p2 = P_{n-1}(z)
            pp = n * (z * p1 - p2) / (z * z - 1.0)  # P'_n(z)
            z1 = z
            z = z1 - p1 / pp
            if abs(z - z1) < 1e-15:
                break

        # Weight
        p1 = 1.0
        p2 = 0.0
        for j in range(1, n + 1):
            p3 = p2
            p2 = p1
            p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j
        pp = n * (z * p1 - p2) / (z * z - 1.0)
        wi = 2.0 / ((1.0 - z * z) * pp * pp)

        # Symmetry: place +/-z
        x[i - 1] = -z
        x[n - i] = z
        w[i - 1] = wi
        w[n - i] = wi

    return x, w


def _beta_from_t(t: float, Ht: float) -> float:
    """
    One-step "turn" contribution beta_{t+1} in radians, expressed via t = m-1 and H_t.
    beta(m) = atan( sqrt(1 - r^2/4) / (r*(t+1/2)) ), where r = sqrt(H_t/t).
    """
    r = math.sqrt(Ht / t)
    q = math.sqrt(1.0 - 0.25 * r * r) / (r * (t + 0.5))
    return math.atan(q)


def _beta_real(m: float) -> float:
    """Continuous extension of beta(m) used for numerical integration in the tail."""
    t = m - 1.0
    return _beta_from_t(t, _harmonic_asymp(t))


def _alpha_from_t(t: float, Ht: float) -> float:
    """alpha_{t+1} = arccos(r/2), where r = sqrt(H_t/t)."""
    r = math.sqrt(Ht / t)
    return math.acos(0.5 * r)


def _integral_beta_log(
    a: float, b: float, gl_x: list[float], gl_w: list[float], seg_u: float = 0.5
) -> float:
    """
    Compute integral_a^b beta(x) dx using a log-substitution x = exp(u):
      integral beta(x) dx = integral beta(exp(u)) * exp(u) du

    The u-interval is split into fixed-size chunks (in log-space) and
    integrated using Gauss-Legendre quadrature per chunk.
    """
    if b <= a:
        return 0.0
    ua = math.log(a)
    ub = math.log(b)
    steps = max(1, int(math.ceil((ub - ua) / seg_u)))
    du = (ub - ua) / steps

    total = 0.0
    for i in range(steps):
        u0 = ua + i * du
        u1 = ua + (i + 1) * du
        mid = 0.5 * (u0 + u1)
        half = 0.5 * (u1 - u0)

        s = 0.0
        for xi, wi in zip(gl_x, gl_w):
            u = mid + half * xi
            m = math.exp(u)
            s += wi * _beta_real(m) * m  # dx = m du
        total += s * half

    return total


class CoinLoopsSolver:
    """
    Computes S(n) = sum_{k=2}^n theta_k (in radians) efficiently, where n can be huge.

    We use:
      S(n) = alpha_n + sum_{m=2}^{n-1} beta_m
    with alpha_n and beta_m depending only on harmonic numbers.
    """

    def __init__(self, M: int = 500_000, quad_order: int = 16) -> None:
        self.M = int(M)
        if self.M < 10_000:
            self.M = 10_000

        self.gl_x, self.gl_w = _gauss_legendre(int(quad_order))

        # Precompute exact harmonic numbers H_t and prefix sums of beta for t <= M.
        self._H = array("d", [0.0]) * (self.M + 1)  # H[t] for integer t
        self._pref_beta = array("d", [0.0]) * (self.M + 1)  # sum_{m=2}^i beta_m

        H = 0.0
        for t in range(1, self.M + 1):
            H += 1.0 / t
            self._H[t] = H

        s = 0.0
        for m in range(2, self.M + 1):
            t = m - 1
            s += _beta_from_t(float(t), self._H[t])
            self._pref_beta[m] = s

    def _harmonic(self, t: int) -> float:
        if t <= self.M:
            return self._H[t]
        return _harmonic_asymp(float(t))

    def _alpha(self, n: int) -> float:
        if n <= 1:
            return 0.0
        t = n - 1
        return _alpha_from_t(float(t), self._harmonic(t))

    def _tail_sum_beta(self, a: int, b: int) -> float:
        """
        Approximate sum_{m=a}^b beta_m using Euler-Maclaurin with an integral:
          sum f(k) ~= integral_a^b f(x) dx + (f(a)+f(b))/2
        The integral is computed in log-space via Gauss-Legendre quadrature.
        """
        if b < a:
            return 0.0

        ia = float(a)
        ib = float(b)
        integ = _integral_beta_log(ia, ib, self.gl_x, self.gl_w)
        fa = _beta_real(ia)
        fb = _beta_real(ib)
        return integ + 0.5 * (fa + fb)

    def rotation_sum(self, n: int) -> float:
        """
        Returns S(n) in radians.
        """
        if n < 2:
            return 0.0

        if n <= self.M + 1:
            # sum_{m=2}^{n-1} beta_m is pref_beta[n-1]
            return self._alpha(n) + self._pref_beta[n - 1]

        return (
            self._alpha(n)
            + self._pref_beta[self.M]
            + self._tail_sum_beta(self.M + 1, n - 1)
        )

    def coins_for_loops(self, loops: int) -> int:
        """
        Smallest n such that S(n) > 2*pi*loops.
        """
        if loops <= 0:
            return 1

        target = 2.0 * math.pi * loops

        lo = 1
        hi = max(2, loops * 5)
        # Find an upper bound
        while self.rotation_sum(hi) <= target:
            hi *= 2

        # Binary search
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if self.rotation_sum(mid) > target:
                hi = mid
            else:
                lo = mid
        return hi


def main() -> None:
    loops = 2020
    if len(sys.argv) >= 2:
        loops = int(sys.argv[1])

    solver = CoinLoopsSolver()

    # Test values given in the problem statement:
    assert solver.coins_for_loops(1) == 31
    assert solver.coins_for_loops(2) == 154
    assert solver.coins_for_loops(10) == 6947

    print(solver.coins_for_loops(loops))


if __name__ == "__main__":
    main()
