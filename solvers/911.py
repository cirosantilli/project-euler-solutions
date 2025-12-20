#!/usr/bin/env python3
"""Project Euler 911 - Khinchin Exceptions

We need the geometric mean of k_∞(ρ_n) for 0 <= n <= 50, where

    ρ_n = sum_{i>=0} 2^n / 2^{2^i} = 2^n * B(2,∞)
    B(u, v) = sum_{k=0..v} 1 / u^{2^k}

We use explicit continued-fraction recurrences proved by Jeffrey Shallit
("Simple Continued Fractions for Some Irrational Numbers", 1979):

* Theorem 11 builds the continued fraction for u^t * B(u,∞) by repeatedly
  appending a fixed middle block and a mirror of the previous coefficients.
  For n>=1 we apply it with u=2, t=n.

* For n=0 (i.e. B(2,∞)) the u=2 special case needs care. We compute ln(k_∞)
  from two large prefixes and use a 1/L extrapolation.

No external libraries are used.
"""

from __future__ import annotations

import math
from typing import List, Tuple


def _seed_rho(n: int) -> Tuple[List[int], str, int]:
    """Return (coeffs, mode, t2).

    coeffs are the continued fraction coefficients of a finite truncation that is a
    prefix-generator for rho_n.

    mode is either:
      - 't1' for Theorem 1 (used only for n==0)
      - 't11' for Theorem 11 (used for n>=1)

    For mode 't11', t2 = 2^n - 1. For 't1', t2 is unused (0).
    """
    if n < 0:
        raise ValueError("n must be non-negative")

    if n == 0:
        # For rho_0 = B(2,∞), start from B(2,1) = [0; 1, 3] and iterate Theorem 1.
        # (Theorem 1 is stated for u>=3, but u=2 works with 0-elimination.)
        return [0, 1, 3], "t1", 0

    # Theorem 11 parameters (u=2, t=n).
    # v' is the least non-negative integer such that 2^{v'} > t.
    vprime = 0
    while (1 << vprime) <= n:
        vprime += 1

    # d = 2^{v'} - t
    d = (1 << vprime) - n

    # For u=2, u^t = 2^n and u^d = 2^d.
    ut = 1 << n
    ud = 1 << d

    # c = u^t * B(u, v'-1). For n>=1 this is an integer:
    # c = sum_{k=0..v'-1} 2^{n-2^k}.
    c = 0
    for k in range(vprime):
        c += 1 << (n - (1 << k))

    # Seed for v = v' + 1 (Theorem 11(A)):
    # 2^n B(2, v'+1) = [c, 2^d - 1, 1, 2^n - 1, 2^d]
    coeffs = [c, ud - 1, 1, ut - 1, ud]
    return coeffs, "t11", ut - 1


def _extend_theorem11(coeffs: List[int], t2: int) -> List[int]:
    """One application of Shallit Theorem 11(B) for u=2.

    Input: coeffs = [a0, a1, ..., an]
    Output (before 0-elimination):
      [a0, ..., an, t2, 1, an-1, a_{n-1}, ..., a1]

    If an==1 then an-1==0 occurs and we apply
      [..., 1, 0, x, ...] -> [..., 1+x, ...]
    """
    if len(coeffs) < 3:
        raise ValueError("coeffs too short")
    last = coeffs[-1]

    if t2 <= 0:
        raise ValueError("t2 must be positive for theorem 11")

    if last > 1:
        tail_rev = coeffs[-2:0:-1]  # a_{n-1}..a1
        return coeffs + [t2, 1, last - 1] + tail_rev

    # last == 1: eliminate the 0 produced by (last-1)
    # Pattern becomes: ..., t2, 1, 0, a_{n-1}, a_{n-2}, ..., a1
    # => ..., t2, 1 + a_{n-1}, a_{n-2}, ..., a1
    prelast = coeffs[-2]
    tail_rev_rest = coeffs[-3:0:-1]  # a_{n-2}..a1
    return coeffs + [t2, 1 + prelast] + tail_rev_rest


def _extend_theorem1(coeffs: List[int]) -> List[int]:
    """One application of Shallit Theorem 1(B) for u=2, with 0-elimination."""
    if len(coeffs) < 3:
        raise ValueError("coeffs too short")
    last = coeffs[-1]

    if last > 1:
        tail_rev = coeffs[-2:0:-1]
        return coeffs[:-1] + [last + 1, last - 1] + tail_rev

    # last == 1: eliminate 0 from (last-1)
    prelast = coeffs[-2]
    tail_rev_rest = coeffs[-3:0:-1]
    return coeffs[:-1] + [2 + prelast] + tail_rev_rest


def rho_cf_prefix(n: int, count: int) -> List[int]:
    """Return the first (count+1) coefficients [a0, a1, ..., a_count] of rho_n."""
    if count < 0:
        raise ValueError("count must be non-negative")

    coeffs, mode, t2 = _seed_rho(n)

    if mode == "t11":
        # Theorem 11 expansions are prefix-stable: the previous coefficients are
        # a literal prefix of the next truncation.
        while len(coeffs) <= count:
            coeffs = _extend_theorem11(coeffs, t2)
        return coeffs[: count + 1]

    # For rho_0 = B(2,∞), Theorem 1 is *not* prefix-stable, but Shallit proves
    # that the first 2^v partial denominators of B(2,v) match B(2,∞). Starting
    # from v=1 and applying Theorem 1 increments v by 1.
    v = 1

    def canonicalize_tail(cf: List[int]) -> None:
        # Finite rationals have two representations if the last term is 1; choose
        # the canonical one with no trailing 1.
        if len(cf) > 1 and cf[-1] == 1:
            cf[-2] += 1
            cf.pop()

    canonicalize_tail(coeffs)
    while (1 << v) < (count + 1):
        coeffs = _extend_theorem1(coeffs)
        canonicalize_tail(coeffs)
        v += 1
    return coeffs[: count + 1]


def _avg_log_positive_ints(arr: List[int]) -> float:
    """Average of ln(a) over positive integers a in arr."""
    s = 0.0
    for a in arr:
        s += math.log(a)
    return s / len(arr)


_LOG_RHO0: float | None = None


def _log_khinchin_rho0() -> float:
    """Compute ln(k_∞(rho_0)) for rho_0 = B(2,∞).

    For u=2, Theorem 1 can introduce many trailing-1 ambiguities in finite
    truncations. We keep the truncations in canonical form and compute the
    average log for two large prefix lengths L, then extrapolate with an O(1/L)
    error model:

        mu(L) = ln(k_L) = mu_∞ + c/L + o(1/L).
    """
    global _LOG_RHO0
    if _LOG_RHO0 is not None:
        return _LOG_RHO0

    coeffs, mode, _ = _seed_rho(0)
    assert mode == "t1"

    def canonicalize_tail(cf: List[int]) -> None:
        if len(cf) > 1 and cf[-1] == 1:
            cf[-2] += 1
            cf.pop()

    canonicalize_tail(coeffs)

    # Build up to two checkpoints (chosen so tail lengths are large and differ
    # by a factor 4, giving a stable extrapolation).
    mu1 = mu2 = 0.0
    L1 = L2 = 0

    for step in range(1, 23):
        coeffs = _extend_theorem1(coeffs)
        canonicalize_tail(coeffs)

        if step == 20:
            tail = coeffs[1:]
            L1 = len(tail)
            mu1 = _avg_log_positive_ints(tail)

        if step == 22:
            tail = coeffs[1:]
            L2 = len(tail)
            mu2 = _avg_log_positive_ints(tail)

    if not (L1 and L2 and L2 > L1):
        raise AssertionError("unexpected checkpoint lengths")

    # Solve mu(L) = mu_inf + c/L using the two points (L1,mu1) and (L2,mu2).
    mu_inf = (mu2 * L2 - mu1 * L1) / (L2 - L1)

    _LOG_RHO0 = mu_inf
    return mu_inf


def _khinchin_log_limit(n: int, max_steps: int = 80) -> float:
    """Return ln(k_∞(rho_n))."""
    if n == 0:
        return _log_khinchin_rho0()

    coeffs, mode, t2 = _seed_rho(n)
    assert mode == "t11"

    tail = coeffs[1:]
    L = len(tail)
    if L < 2:
        raise AssertionError("unexpectedly short tail")

    # Exact average ln(a_i) over this finite prefix.
    mu = _avg_log_positive_ints(tail)

    # We only need the first two and last two tail digits to update the mean,
    # because Theorem 11(B) appends a fixed middle block and then a mirror.
    first = tail[0]
    second = tail[1]
    last = tail[-1]
    prelast = tail[-2]

    for _ in range(max_steps):
        mu_old = mu
        if last > 1:
            L_new = 2 * L + 2
            delta = -math.log(last) + math.log(t2) + math.log(last - 1)
            mu = mu * ((2 * L) / L_new) + delta / L_new
        else:
            # last == 1: 1,0,a_{n-1} -> 1+a_{n-1}
            L_new = 2 * L
            delta = -math.log(prelast) + math.log(t2) + math.log(prelast + 1)
            mu = mu + delta / L_new

        L = L_new
        prelast = second
        last = first

        if abs(mu - mu_old) < 1e-15:
            break

    return mu


def k_infty(n: int) -> float:
    """Compute k_infty(rho_n)."""
    return math.exp(_khinchin_log_limit(n))


def solve() -> str:
    """Return the required answer as a formatted string."""
    # Assertions from the problem statement.
    assert rho_cf_prefix(2, 7) == [3, 3, 1, 3, 4, 3, 1, 3]
    assert round(k_infty(2), 6) == 2.059767

    # Geometric mean of k_infty(rho_n) for 0<=n<=50 is exp(avg log).
    total_log = 0.0
    for n in range(51):
        total_log += _khinchin_log_limit(n)
    avg_log = total_log / 51.0

    ans = math.exp(avg_log)
    return f"{ans:.6f}"


if __name__ == "__main__":
    print(solve())
