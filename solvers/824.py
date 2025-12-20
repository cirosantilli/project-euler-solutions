#!/usr/bin/env python3
"""
Project Euler 824: Chess Sliders

A Slider moves one square left/right on an N×N cylindrical board (wrap-around).
Sliders attack if adjacent horizontally in the same row.

Let L(N,K) be the number of ways to place K non-attacking sliders.
Compute L(10^9, 10^15) mod (10^7+19)^2.

No external libraries are used.
"""

from __future__ import annotations

from array import array
import math


# ----------------------------
# Small reference computation
# ----------------------------


def _cycle_independence_poly_coeffs(n: int) -> list[int]:
    """
    Independence polynomial of an n-cycle (adjacent attack):
        P_n(x) = sum_{r=0..floor(n/2)} a_r x^r

    Closed form for r>=1:
        a_r = n/(n-r) * C(n-r, r)
    """
    max_r = n // 2
    a = [0] * (max_r + 1)
    a[0] = 1
    for r in range(1, max_r + 1):
        a[r] = n * math.comb(n - r, r) // (n - r)
    return a


def L_small(n: int, k: int) -> int:
    """
    Compute L(n,k) exactly for small n,k:
        L(n,k) = [x^k] (P_n(x))^n
    """
    base = _cycle_independence_poly_coeffs(n)
    res = [1]
    for _ in range(n):
        new = [0] * min(k + 1, len(res) + len(base) - 1)
        for i, ai in enumerate(res):
            if ai == 0:
                continue
            for j, bj in enumerate(base):
                ij = i + j
                if ij > k:
                    break
                new[ij] += ai * bj
        res = new
    return res[k] if k < len(res) else 0


# Asserts from the problem statement examples
assert L_small(2, 2) == 4
assert L_small(6, 12) == 4204761


# ----------------------------
# Main solve
# ----------------------------


def solve() -> int:
    p = 10_000_019
    mod = p * p

    N = 10**9
    K = 10**15
    t_max = K // N  # 1_000_000

    # Inverses mod p
    invp = array("I", [0]) * p
    invp[1] = 1
    for i in range(2, p):
        invp[i] = (p - (p // i) * invp[p % i] % p) % p

    # factorial mod p^2 for 0..p-1
    fac = array("Q", [0]) * p
    f = 1
    fac[0] = 1
    for i in range(1, p):
        f = (f * i) % mod
        fac[i] = f

    # harmonic numbers H[v] = sum_{i<=v} 1/i mod p
    H = array("I", [0]) * p
    h = 0
    for i in range(1, p):
        h += invp[i]
        h %= p
        H[i] = h

    # Wilson quotient: (p-1)! ≡ -1 + p*w (mod p^2)
    F = fac[p - 1]
    w = ((F + 1) // p) % p

    def inv_mod_p2(a: int) -> int:
        a %= mod
        r = a % p
        x = invp[r]
        return (x * (2 - (a * x) % mod)) % mod

    def Fpow(u: int) -> int:
        um = u % p
        tmp = (1 - (um * p % mod) * w % mod) % mod
        if u & 1:
            tmp = (-tmp) % mod
        return tmp

    def unit_factorial(n: int) -> int:
        res = 1
        while n:
            u, v = divmod(n, p)
            res = (res * fac[v]) % mod
            corr = (u % p) * H[v] % p
            res = (res * (1 + corr * p)) % mod
            res = (res * Fpow(u)) % mod
            n = u
        return res

    def vp_fact(n: int) -> int:
        q = n // p
        return q + q // p  # exact here since all inputs < p^3

    def binom_mod_p2(n: int, k: int) -> int:
        if k < 0 or k > n:
            return 0
        nk = n - k
        e = vp_fact(n) - vp_fact(k) - vp_fact(nk)
        if e >= 2:
            return 0

        un = unit_factorial(n)
        uk = unit_factorial(k)
        unk = unit_factorial(nk)

        val = (un * inv_mod_p2(uk)) % mod
        val = (val * inv_mod_p2(unk)) % mod
        if e == 1:
            val = (val * p) % mod
        return val

    def coeff_alpha_power(M: int, d: int) -> int:
        if d == 0:
            return 1
        B = binom_mod_p2(M - d - 1, d - 1)
        return ((M % mod) * inv_mod_p2(d % mod) % mod) * B % mod

    comb = 1  # C(N,0)
    ans = 0

    d = K
    M = N * N

    for t in range(0, t_max + 1):
        if t:
            comb = (comb * ((N - t + 1) % mod)) % mod
            comb = (comb * inv_mod_p2(t)) % mod

        ans = (ans + comb * coeff_alpha_power(M, d)) % mod

        d -= N
        M -= 2 * N

    return ans


if __name__ == "__main__":
    print(solve())
