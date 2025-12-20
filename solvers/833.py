#!/usr/bin/env python3
"""
Project Euler 833 — Square Triangle Products

Triangle numbers: T(k) = k(k+1)/2.

We define S(N) as the sum of all integers c (0 < c <= N) such that there exist
integers 0 < a < b with:

    c^2 = T(a) * T(b)

We must compute S(10^35) mod 136101521.

This implementation:
  • Uses a Pell/Lucas parametrisation (via OEIS A322699 style identities)
  • Enumerates solutions uniquely using coprime index pairs
  • Sums polynomial families via finite differences and binomial identities
  • Avoids modular inverses (modulus may be composite)
  • Includes asserts for all test values in the statement
"""

import math
import sys

MOD = 136101521


def lucas_U_pair(n, i, j):
    """
    For fixed n define P = 4n + 2 and Lucas U-sequence:

        U_0 = 0
        U_1 = 1
        U_k = P*U_{k-1} - U_{k-2}

    Return (U_i, U_j).
    """
    P = 4 * n + 2
    kmax = i if i > j else j

    if kmax == 0:
        return (0, 0)
    if kmax == 1:
        return (1 if i == 1 else 0, 1 if j == 1 else 0)

    u0, u1 = 0, 1
    ui = 1 if i == 1 else None
    uj = 1 if j == 1 else None

    for k in range(2, kmax + 1):
        u0, u1 = u1, P * u1 - u0
        if k == i:
            ui = u1
        if k == j:
            uj = u1

    return ui, uj


def c_value(n, i, j):
    """Compute c(n,i,j) = T(n) * U_i * U_j where T(n)=n(n+1)/2."""
    t = n * (n + 1) // 2
    ui, uj = lucas_U_pair(n, i, j)
    return t * ui * uj


def max_n_for_pair(i, j, N):
    """
    For fixed coprime (i,j), c_value(n,i,j) is strictly increasing for n>=1.
    Find largest n such that c_value(n,i,j) <= N.
    """
    lo, hi = 0, 1
    while c_value(hi, i, j) <= N:
        hi *= 2
    lo = hi // 2

    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if c_value(mid, i, j) <= N:
            lo = mid
        else:
            hi = mid
    return lo


def forward_differences(values):
    """
    Given values f(0), f(1), ..., f(d) for a polynomial f of degree <= d,
    return the Newton-series coefficients:

        a_m = Δ^m f(0)

    so that:
        f(n) = sum_{m=0..d} a_m * C(n, m)
    """
    coeffs = []
    cur = values[:]
    while cur:
        coeffs.append(cur[0])
        cur = [cur[k + 1] - cur[k] for k in range(len(cur) - 1)]
    return coeffs


def binom_exact(n, k):
    """Exact binomial coefficient C(n,k) computed with integer arithmetic."""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    if k > n - k:
        k = n - k
    res = 1
    for t in range(1, k + 1):
        res = res * (n - k + t) // t
    return res


def sum_poly_prefix(f0_to_d, M):
    """
    Given f(0..d) for polynomial f, compute sum_{n=0..M} f(n) exactly.

    Using Newton series:
      f(n) = sum a_m * C(n,m)
      sum_{n=0..M} C(n,m) = C(M+1, m+1)
    """
    a = forward_differences(f0_to_d)
    Mp1 = M + 1
    total = 0
    for m in range(len(a)):
        total += a[m] * binom_exact(Mp1, m + 1)
    return total


def maxJ_for_N(N):
    """
    Determine maximum j such that c(1,1,j) <= N.
    For n=1, T(1)=1 and P=6, so c(1,1,j)=U_j(6).
    """
    P = 6
    u0, u1 = 0, 1
    maxj = 1
    for k in range(2, 400):
        u0, u1 = u1, P * u1 - u0
        if u1 > N:
            break
        maxj = k
    return maxj


def S_exact(N):
    """
    Compute S(N) exactly.

    Key uniqueness rule:
      any triple corresponds to exactly one (n, i<j) with gcd(i,j)=1.
    """
    maxj = maxJ_for_N(N)
    total = 0

    for i in range(1, maxj):
        for j in range(i + 1, maxj + 1):
            if math.gcd(i, j) != 1:
                continue
            if c_value(1, i, j) > N:
                continue

            M = max_n_for_pair(i, j, N)
            if M <= 0:
                continue

            deg = i + j
            fvals = [c_value(n, i, j) for n in range(deg + 1)]
            total += sum_poly_prefix(fvals, M)

    return total


def S_mod(N, mod=MOD):
    return S_exact(N) % mod


def main():
    # Asserts for test values given in the statement
    assert S_exact(100) == 155
    assert S_exact(10**5) == 1479802
    assert S_exact(10**9) == 241614948794

    N = 10**35
    print(S_mod(N, MOD))


if __name__ == "__main__":
    main()
