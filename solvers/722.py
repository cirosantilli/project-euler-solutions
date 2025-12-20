#!/usr/bin/env python3
"""
Project Euler 722 - Slowly Converging Series

We need E_k(q) = sum_{n>=1} sigma_k(n) q^n, where sigma_k(n) = sum_{d|n} d^k.
The target is E_15(1 - 1/2^25), reported in scientific notation with
12 digits after the decimal point.

No external libraries are used.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import List


def bernoulli_numbers(n: int) -> List[Fraction]:
    """
    Compute Bernoulli numbers B_0..B_n exactly as Fractions using the
    Akiyama–Tanigawa algorithm.
    """
    a = [Fraction(0) for _ in range(n + 1)]
    b = [Fraction(0) for _ in range(n + 1)]
    for m in range(n + 1):
        a[m] = Fraction(1, m + 1)
        for j in range(m, 0, -1):
            a[j - 1] = j * (a[j - 1] - a[j])
        b[m] = a[0]
    return b


def sigma_k_sieve(n_max: int, k: int) -> List[int]:
    """
    Sieve sigma_k(n) for n=0..n_max in O(n log n).
    Returns a list where res[n] == sigma_k(n).
    """
    res = [0] * (n_max + 1)
    for d in range(1, n_max + 1):
        dk = d**k
        for m in range(d, n_max + 1, d):
            res[m] += dk
    return res


def E_direct(k: int, q: float, tol: float = 1e-15) -> float:
    """
    Directly sum E_k(q) = sum_{n>=1} sigma_k(n) q^n with an adaptive cutoff.
    Intended for cases where q is not extremely close to 1, or where only
    modest accuracy is required (here: rounding to 12 decimals in sci notation).

    tol controls the stopping threshold relative to the running partial sum.
    """
    if not (0.0 < q < 1.0):
        raise ValueError("q must be in (0, 1)")

    total = 0.0
    n = 1
    # Start with a modest sieve and grow if needed.
    n_max = 512
    sig = sigma_k_sieve(n_max, k)
    qpow = q

    # Require a minimum n before allowing termination so we don't stop too early.
    min_n = 50

    while True:
        if n > n_max:
            n_max *= 2
            sig = sigma_k_sieve(n_max, k)

        term = float(sig[n]) * qpow
        total += term

        # Termination check: current term is tiny relative to total and stays tiny.
        scale = max(1.0, abs(total))
        if n >= min_n and abs(term) < tol * scale:
            # Verify the next few terms are also below threshold.
            ok = True
            qpow2 = qpow * q
            for j in range(1, 6):
                nn = n + j
                if nn > n_max:
                    n_max *= 2
                    sig = sigma_k_sieve(n_max, k)
                t2 = float(sig[nn]) * qpow2
                if abs(t2) >= tol * scale:
                    ok = False
                    break
                qpow2 *= q
            if ok:
                break

        n += 1
        qpow *= q

    return total


def E_odd_modular(k: int, q: float) -> float:
    """
    Compute E_k(q) for odd k>=3 using the modular transformation of the
    Eisenstein series of weight k+1 (which is >=4 here).

    Let k = 2r-1 with r>=2. Write q = exp(-2*pi*t) where t = -ln(q)/(2*pi).
    Then:
        E_k(q) = t^(-2r) * E_k(exp(-2*pi/t)) + (t^(-2r) - 1) * C_r
    where:
        C_r = (2r-1)! * zeta(2r) / (2*pi)^(2r)
            = (-1)^(r+1) * B_{2r} / (4r)
    and B_{2r} is the Bernoulli number.

    For the values in this problem, exp(-2*pi/t) is astronomically small,
    so the t^(-2r)*E_k(exp(-2*pi/t)) term is negligible at 12-decimal rounding,
    but we still include it when it is numerically representable.
    """
    if k % 2 == 0 or k < 3:
        raise ValueError("k must be odd and >= 3")
    if not (0.0 < q < 1.0):
        raise ValueError("q must be in (0, 1)")

    r = (k + 1) // 2
    b = bernoulli_numbers(2 * r)[2 * r]
    c_r = ((-1) ** (r + 1)) * b / (4 * r)  # exact Fraction

    # Use log1p for better accuracy when q is close to 1.
    if 1.0 - q < 1e-2:
        t = -math.log1p(-(1.0 - q)) / (2.0 * math.pi)
    else:
        t = -math.log(q) / (2.0 * math.pi)
    t_pow = t ** (-2 * r)

    # Compute q2 = exp(-2*pi/t) only if it won't underflow to 0.0.
    exponent = (2.0 * math.pi) / t
    if exponent > 745.0:  # exp(-x) underflows to 0 for x ~ 745 in IEEE doubles
        e_small = 0.0
    else:
        q2 = math.exp(-exponent)
        # q2 is extremely small; direct summation converges in very few steps.
        e_small = E_direct(k, q2, tol=1e-18)

    return t_pow * e_small + (t_pow - 1.0) * float(c_r)


def to_sci_12(x: float) -> str:
    """
    Scientific notation with 12 digits after decimal point, and an exponent
    with no leading '+' and no leading zeros (matches the problem statement style).
    """
    s = f"{x:.12e}"  # e.g. '3.872...e+02'
    mant, exp = s.split("e")
    return f"{mant}e{int(exp)}"


def compute_E(k: int, q: float) -> float:
    """Dispatcher for the few k-values relevant to this problem."""
    if k == 1:
        return E_direct(1, q, tol=1e-16)
    if k % 2 == 1 and k >= 3:
        return E_odd_modular(k, q)
    raise ValueError("This solver is intended for odd k (k=1 or k>=3).")


def main() -> None:
    # Test values from the problem statement (scientific notation, 12 decimals).
    examples = [
        (1, 4, "3.872155809243e2"),
        (3, 8, "2.767385314772e10"),
        (7, 15, "6.725803486744e39"),
    ]
    for k, p, expected in examples:
        q = 1.0 - 1.0 / (2.0**p)
        got = to_sci_12(compute_E(k, q))
        assert got == expected, (k, p, got, expected)

    # Target
    q_target = 1.0 - 1.0 / (2.0**25)
    ans = compute_E(15, q_target)
    print(to_sci_12(ans))


if __name__ == "__main__":
    main()
