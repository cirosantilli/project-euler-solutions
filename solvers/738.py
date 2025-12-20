#!/usr/bin/env python3
"""
Project Euler 738: Counting Ordered Factorisations

We count tuples (x1 <= x2 <= ... <= xk) of positive integers whose product is <= N.
Because 1's can only appear at the beginning of a nondecreasing tuple, we can:
- count all "base tuples" with factors >= 2
- then account for any number of leading 1's via a simple weight.

The program prints D(10^10, 10^10) mod 1_000_000_007.
"""

from __future__ import annotations

import math
from functools import lru_cache

MOD = 1_000_000_007


def _isqrt(n: int) -> int:
    return math.isqrt(n)


def _icbrt_floor(n: int) -> int:
    """floor(cuberoot(n)) for n >= 0, using math.cbrt + correction."""
    x = int(round(math.cbrt(n)))
    # correct rounding errors
    while (x + 1) * (x + 1) * (x + 1) <= n:
        x += 1
    while x * x * x > n:
        x -= 1
    return x


def _sum_floor_range(m: int, l: int, r: int) -> int:
    """Return sum_{i=l..r} floor(m / i) in O(#distinct quotients)."""
    res = 0
    i = l
    while i <= r:
        q = m // i
        j = m // q
        if j > r:
            j = r
        res += q * (j - i + 1)
        i = j + 1
    return res


def _sum_arith(l: int, r: int) -> int:
    """Return sum_{i=l..r} i."""
    n = r - l + 1
    return (l + r) * n // 2


@lru_cache(maxsize=None)
def _count_and_length(m: int, a: int) -> tuple[int, int]:
    """
    Count base tuples and sum of their lengths:

    - tuples have length >= 1
    - all factors are integers >= a (and we will call with a >= 2)
    - nondecreasing (enforced by passing the last chosen factor as 'a')
    - product <= m

    Returns (C, L) modulo MOD where:
      C = number of tuples
      L = sum of lengths over all tuples
    """
    if m < a:
        return 0, 0

    aa = a * a
    # only length-1 tuples if even a*a > m
    if aa > m:
        cnt = (m - a + 1) % MOD
        return cnt, cnt

    # length-1 tuples: choose a single factor in [a..m]
    C = m - a + 1
    L = C

    s = _isqrt(m)

    # If a^3 > m then every f in [a..s] is > cbrt(m), so the subproblem is always a base case.
    if aa * a > m:
        l = a
        if l <= s:
            # for each f: subC = floor(m/f) - f + 1, and length contribution is 2*subC
            sf = _sum_floor_range(m, l, s)
            sa = _sum_arith(l, s)
            baseC = sf - sa + (s - l + 1)
            C += baseC
            L += 2 * baseC
        return C % MOD, L % MOD

    t = _icbrt_floor(m)
    upto = t if t < s else s

    # recursive part: f in [a..upto]
    if upto >= a:
        for f in range(a, upto + 1):
            subC, subL = _count_and_length(m // f, f)
            C += subC
            L += subL + subC

    # base part: f in [max(a, upto+1)..s]
    l = a if a > upto + 1 else (upto + 1)
    if l <= s:
        sf = _sum_floor_range(m, l, s)
        sa = _sum_arith(l, s)
        baseC = sf - sa + (s - l + 1)
        C += baseC
        L += 2 * baseC

    return C % MOD, L % MOD


def D(N: int, K: int) -> int:
    """
    Compute D(N, K) modulo MOD for the regime used in the problem (K large).

    For n = 1: d(1,k)=1 for all k, so contribution is K.
    For n > 1: any factorisation can be represented by a "base tuple" of factors >= 2,
    plus any number of leading 1's. For a base tuple of length ℓ, there are (K-ℓ+1) ways
    to pad with 1's to reach any total length <= K.

    If K >= floor(log2(N)), then every base tuple length ℓ that can appear satisfies ℓ <= K,
    so we don't need to truncate by length.
    """
    if N <= 0 or K <= 0:
        return 0
    max_len = N.bit_length() - 1  # floor(log2 N)
    if K < max_len:
        raise ValueError("This implementation assumes K >= floor(log2 N).")

    _count_and_length.cache_clear()
    C, L = _count_and_length(N, 2)

    # Total = K (for n=1) + sum_{base tuples} (K - len + 1)
    #        = K + (K+1)*C - L
    ans = (K % MOD + ((K + 1) % MOD) * C - L) % MOD
    return ans


def main() -> None:
    # Given test values in the statement:
    assert D(10, 10) == 153
    assert D(100, 100) == 35384

    N = 10**10
    K = 10**10
    print(D(N, K))


if __name__ == "__main__":
    main()
