#!/usr/bin/env python3
"""
Project Euler 756: Approximating a Sum

We approximate S = sum_{k=1..n} f(k) using a random increasing m-tuple
0 = X_0 < X_1 < ... < X_m <= n:

    S* = sum_{i=1..m} f(X_i) (X_i - X_{i-1})

and define the error Δ = S - S*.

This program computes E(Δ | f(k), n, m) exactly (no simulation), and then
evaluates it for f(k) = φ(k) (Euler's totient), for the problem's n and m.

No external libraries are used (stdlib only).
"""

from array import array


def totients_up_to(n: int) -> array:
    """
    Compute phi[k] = φ(k) for 0 <= k <= n using Euler's linear sieve.
    Returns an array('I') of length n+1.
    """
    phi = array("I", [0]) * (n + 1)
    if n >= 1:
        phi[1] = 1

    primes = []  # list of primes discovered so far
    primes_append = primes.append
    phi_local = phi

    for i in range(2, n + 1):
        if phi_local[i] == 0:  # i is prime
            primes_append(i)
            phi_local[i] = i - 1
        for p in primes:
            ip = i * p
            if ip > n:
                break
            if i % p == 0:
                phi_local[ip] = phi_local[i] * p
                break
            else:
                phi_local[ip] = phi_local[i] * (p - 1)
    return phi


def _kahan_add(total: float, c: float, x: float) -> tuple[float, float]:
    """One step of Kahan compensated summation."""
    y = x - c
    t = total + y
    c = (t - total) - y
    return t, c


def cutoff_index(n: int, m: int, eps: float = 5e-8) -> int:
    """
    Find a safe truncation index K (<= n-m) such that the remaining tail
    contribution is provably < eps.

    Uses:
      φ(k) <= k <= n, and weights w_k are non-increasing, hence
      sum_{k>K} φ(k) w_k <= n * (n-m-K) * w_{K+1}.

    The returned K is the largest index we still need to include so that
    the tail bound after K is < eps.
    """
    limit = n - m
    if limit <= 0:
        return 0

    w = (n - m) / n  # w_1
    for k in range(1, limit + 1):
        remaining = limit - k
        nk = n - k
        if nk <= m:
            w_next = 0.0
        else:
            w_next = w * (nk - m) / nk

        # Tail after processing k:
        if n * remaining * w_next < eps:
            return k

        w = w_next

    return limit


def expected_error_for_k(n: int, m: int) -> float:
    """
    Expected error when f(k) = k, computed exactly (no truncation).

    Formula:
      E(Δ) = sum_{k=1..n-m} k * C(n-k, m) / C(n, m)
    with weights updated by a simple recurrence (no big binomials).
    """
    limit = n - m
    if limit <= 0:
        return 0.0

    w = (n - m) / n  # w_1 = C(n-1,m)/C(n,m)
    total = 0.0
    c = 0.0

    for k in range(1, limit + 1):
        total, c = _kahan_add(total, c, k * w)

        nk = n - k
        if nk <= m:
            break
        w *= (nk - m) / nk

    return total


def expected_error_for_phi(n: int, m: int, *, truncate: bool = False) -> float:
    """
    Expected error when f(k) = φ(k).

    If truncate=True, a rigorous tail bound is used to compute only the
    necessary initial part of the sum (much faster for the problem's n,m).
    """
    limit = n - m
    if limit <= 0:
        return 0.0

    if truncate:
        upto = cutoff_index(n, m)
        if upto > limit:
            upto = limit
    else:
        upto = limit

    phi = totients_up_to(upto)
    phi_local = phi

    w = (n - m) / n  # w_1
    total = 0.0
    c = 0.0

    for k in range(1, upto + 1):
        total, c = _kahan_add(total, c, float(phi_local[k]) * w)

        nk = n - k
        if nk <= m:
            break
        w *= (nk - m) / nk

    return total


def run_tests() -> None:
    # From the problem statement:
    # E(Δ | k, 100, 50) = 2525/1326 ≈ 1.904223
    v1 = expected_error_for_k(100, 50)
    assert abs(v1 - (2525 / 1326)) < 1e-12
    assert f"{v1:.6f}" == "1.904223"

    # From the problem statement:
    # E(Δ | φ(k), 10^4, 10^2) ≈ 5842.849907
    v2 = expected_error_for_phi(10_000, 100, truncate=False)
    assert f"{v2:.6f}" == "5842.849907"


def solve() -> None:
    n = 12_345_678
    m = 12_345
    ans = expected_error_for_phi(n, m, truncate=True)
    print(f"{ans:.6f}")


if __name__ == "__main__":
    run_tests()
    solve()
