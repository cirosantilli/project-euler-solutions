#!/usr/bin/env python3
"""
Project Euler 735: Divisors of 2n^2

We need F(N) = sum_{n=1..N} f(n), where
f(n) = #{ d : d | (2*n^2) and d <= n }.

This solution avoids external libraries and is designed for N=10^12.
"""

from __future__ import annotations

from array import array
from math import isqrt
from typing import Dict


def icbrt(n: int) -> int:
    """Integer cube root: floor(n^(1/3)) for n>=0."""
    if n <= 0:
        return 0
    x = int(round(n ** (1.0 / 3.0)))
    # Fix rounding errors
    while (x + 1) * (x + 1) * (x + 1) <= n:
        x += 1
    while x * x * x > n:
        x -= 1
    return x


def mobius_sieve(n: int) -> array:
    """Return mu[0..n] as array('i') using a linear sieve."""
    mu = array("i", [0]) * (n + 1)
    mu[1] = 1
    primes: list[int] = []
    is_comp = bytearray(n + 1)

    for i in range(2, n + 1):
        if not is_comp[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            ip = i * p
            if ip > n:
                break
            is_comp[ip] = 1
            if i % p == 0:
                mu[ip] = 0
                break
            mu[ip] = -mu[i]
    return mu


def build_odd_squarefree_prefix(K: int) -> array:
    """
    Build prefix count C[x] = # { u <= x : u odd and squarefree }, for 0<=x<=K.
    Returns array('I') length K+1.
    """
    if K < 0:
        raise ValueError("K must be non-negative")

    # is_sf[i] = 1 iff i is odd squarefree
    is_sf = bytearray(b"\x01") * (K + 1)
    is_sf[0] = 0

    # Remove evens
    is_sf[0::2] = b"\x00" * ((K // 2) + 1)

    # Sieve primes up to sqrt(K) (small)
    r = isqrt(K)
    is_prime = bytearray(b"\x01") * (r + 1)
    if r >= 0:
        is_prime[0] = 0
    if r >= 1:
        is_prime[1] = 0
    for i in range(2, isqrt(r) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start : r + 1 : step] = b"\x00" * (((r - start) // step) + 1)

    # Mark multiples of p^2 as non-squarefree
    for p in range(3, r + 1, 2):  # odd primes only (evens already excluded)
        if is_prime[p]:
            sq = p * p
            for m in range(sq, K + 1, sq):
                is_sf[m] = 0

    # Prefix count
    pref = array("I", [0]) * (K + 1)
    c = 0
    for i in range(1, K + 1):
        c += is_sf[i]
        pref[i] = c
    return pref


def brute_f(n: int) -> int:
    """Brute force f(n) for small n (used only for tests)."""
    m = 2 * n * n
    cnt = 0
    for d in range(1, n + 1):
        if m % d == 0:
            cnt += 1
    return cnt


def brute_F(N: int) -> int:
    """Brute force F(N) for small N (used only for tests)."""
    return sum(brute_f(n) for n in range(1, N + 1))


def compute_F(N: int, K_max: int = 20_000_000) -> int:
    """
    Fast computation of F(N) using:
      - a large prefix table for odd squarefree counts up to K
      - a cube-root accelerated Möbius-based routine for larger arguments
      - a product-splitting trick that avoids calling the expensive routine too often.

    Designed primarily for N = 10^12.
    """
    if N < 1:
        return 0

    # Heuristic K: big enough so that N//K is small, but capped for memory.
    # Use integer arithmetic: N^(2/3) = cbrt(N^2).
    K_est = icbrt(N * N)
    K = min(K_max, max(10_000, K_est))
    C_small = build_odd_squarefree_prefix(K)

    twoN = 2 * N
    mu_limit = isqrt(twoN)
    mu = mobius_sieve(mu_limit)

    # Prefix sums of mu over odd indices: M_odd[x] = sum_{1<=k<=x, k odd} mu[k]
    M_odd = array("i", [0]) * (mu_limit + 1)
    s = 0
    for i in range(1, mu_limit + 1):
        if i & 1:
            s += mu[i]
        M_odd[i] = s

    A_cache: Dict[int, int] = {}

    def A_odd(y: int) -> int:
        """
        A_odd(y) = sum_{k odd, k^2<=y} mu(k) * floor(y / k^2)
        Computed in ~O(y^(1/3)) using grouping by floor(y/k^2).
        """
        if y <= 0:
            return 0
        got = A_cache.get(y)
        if got is not None:
            return got

        r = isqrt(y)
        T = icbrt(y)
        if T > r:
            T = r

        total = 0
        # small k directly
        for k in range(1, T + 1, 2):
            total += mu[k] * (y // (k * k))

        # large k grouped by v = y//k^2
        if T < r:
            upper_v = y // ((T + 1) * (T + 1))
            for v in range(1, upper_v + 1):
                hi = isqrt(y // v)
                lo = isqrt(y // (v + 1)) + 1
                if lo <= T:
                    lo = T + 1
                if lo <= hi:
                    total += v * (M_odd[hi] - M_odd[lo - 1])

        A_cache[y] = total
        return total

    C_cache: Dict[int, int] = {}

    def C_odd_squarefree(x: int) -> int:
        """# odd squarefree <= x."""
        if x <= K:
            return int(C_small[x])
        got = C_cache.get(x)
        if got is not None:
            return got
        # odd squarefree count:
        # sum_{k odd} mu(k) * (floor(x/k^2) - floor(x/(2k^2)))
        # = A_odd(x) - A_odd(x//2)
        val = A_odd(x) - A_odd(x // 2)
        C_cache[x] = val
        return val

    # After cancellation:
    #   F(N) = N + S0 + S1
    # where
    #   S0 = sum_{q < t, q*t<=N} C_odd_squarefree(N//(q*t))
    #   S1 = sum_{q even, q < t, q*t<=2N} C_odd_squarefree((2N)//(q*t))
    LIM0 = N // K
    LIM1 = twoN // K

    # ---- Small-product part of S0 (p <= LIM0) ----
    divcnt = array("I", [0]) * (LIM0 + 1)
    for i in range(1, LIM0 + 1):
        for j in range(i, LIM0 + 1, i):
            divcnt[j] += 1

    S0_small = 0
    for p in range(1, LIM0 + 1):
        d = int(divcnt[p])
        r = isqrt(p)
        is_sq = 1 if r * r == p else 0
        pairs = (d - is_sq) // 2
        if pairs:
            S0_small += pairs * C_odd_squarefree(N // p)

    # ---- Large-product part of S0 (p > LIM0) ----
    S0_large = 0
    q_max = isqrt(N)
    for q in range(1, q_max + 1):
        t = q + 1
        lim = LIM0 // q + 1
        if lim > t:
            t = lim
        t_end = N // q
        if t > t_end:
            continue
        while t <= t_end:
            v = N // (q * t)
            t2 = N // (q * v)
            if t2 > t_end:
                t2 = t_end
            S0_large += (t2 - t + 1) * int(C_small[v])
            t = t2 + 1

    # ---- Small-product part of S1 (p <= LIM1, q even) ----
    coeff_even = array("I", [0]) * (LIM1 + 1)
    for q in range(2, LIM1 + 1, 2):
        t_end = LIM1 // q
        if t_end > q:
            for t in range(q + 1, t_end + 1):
                coeff_even[q * t] += 1

    S1_small = 0
    for p in range(1, LIM1 + 1):
        c = int(coeff_even[p])
        if c:
            S1_small += c * C_odd_squarefree(twoN // p)

    # ---- Large-product part of S1 (p > LIM1, q even) ----
    S1_large = 0
    q_max2 = isqrt(twoN)
    for q in range(2, q_max2 + 1, 2):
        t = q + 1
        lim = LIM1 // q + 1
        if lim > t:
            t = lim
        t_end = twoN // q
        if t > t_end:
            continue
        while t <= t_end:
            v = twoN // (q * t)
            t2 = twoN // (q * v)
            if t2 > t_end:
                t2 = t_end
            S1_large += (t2 - t + 1) * int(C_small[v])
            t = t2 + 1

    return N + (S0_small + S0_large) + (S1_small + S1_large)


def main() -> None:
    # Tests from the problem statement
    assert brute_F(15) == 63
    assert brute_F(1000) == 15066

    N = 10**12
    ans = compute_F(N)
    print(ans)


if __name__ == "__main__":
    main()
