#!/usr/bin/env python3
"""
Project Euler 728 — Circle of Coins

No external libraries are used (only Python's standard library).
"""

from array import array
from math import gcd
import sys


MOD = 1_000_000_007


def _pow2_table(n: int) -> array:
    """pow2[i] = 2^i mod MOD for i=0..n (stored as unsigned 32-bit)."""
    pow2 = array("I", [0]) * (n + 1)
    v = 1
    pow2[0] = 1
    for i in range(1, n + 1):
        v <<= 1
        if v >= MOD:
            v -= MOD
        pow2[i] = v
    return pow2


def _totients(n: int) -> array:
    """phi[i] = Euler's totient for i=0..n (stored as unsigned 32-bit)."""
    phi = array("I", [0]) * (n + 1)
    if n >= 1:
        phi[1] = 1

    is_comp = bytearray(n + 1)
    primes = []

    # Linear sieve
    for i in range(2, n + 1):
        if not is_comp[i]:
            primes.append(i)
            phi[i] = i - 1
        for p in primes:
            ip = i * p
            if ip > n:
                break
            is_comp[ip] = 1
            if i % p == 0:
                phi[ip] = phi[i] * p
                break
            else:
                phi[ip] = phi[i] * (p - 1)

    return phi


def _inv_two_pow_minus_one(pow2: array) -> array:
    """
    inv[t] = (2^t - 1)^(-1) mod MOD for t=1..len(pow2)-2, inv[0]=0.

    Built in O(n) time using the "single inversion" trick:
    invert the product once, then back-propagate inverses.
    Handles the (unlikely) case 2^t == 1 (mod MOD) by setting inv[t]=0.
    """
    n = len(pow2) - 1  # pow2 has indices 0..n
    inv = array("I", [0]) * n  # indices 0..n-1 are used for t=0..n-1
    if n <= 1:
        return inv

    # inv[t] temporarily stores prefix products.
    inv[0] = 1
    prod = 1
    for t in range(1, n):
        den = (pow2[t] - 1) % MOD
        if den:
            prod = (prod * den) % MOD
        inv[t] = prod

    inv_prod = pow(prod, MOD - 2, MOD)

    suffix = 1
    for t in range(n - 1, 0, -1):
        den = (pow2[t] - 1) % MOD
        if den:
            # inv[t] still holds prefix product up to t, inv[t-1] is prefix up to t-1.
            inv_t = (inv_prod * inv[t - 1]) % MOD
            inv_t = (inv_t * suffix) % MOD
            inv[t] = inv_t
            suffix = (suffix * den) % MOD
        else:
            # 2^t == 1 (mod MOD) -> denominator 0, inverse not used (ratio=1 handled separately)
            inv[t] = 0

    inv[0] = 0
    return inv


def solve(N: int) -> int:
    """
    Compute S(N) = sum_{n=1..N} sum_{k=1..n} F(n,k) (mod MOD),
    where F(n,k) counts solvable states of the circle-of-coins game.
    """
    pow2 = _pow2_table(N)
    phi = _totients(N)
    inv_den = _inv_two_pow_minus_one(pow2)

    inv2 = (MOD + 1) // 2
    ans = 0

    for m in range(1, N + 1):
        L = N // m
        t = m - 1  # step in exponent

        # A(m) coefficient after summing over k with gcd structure:
        #   A(m) = 2*phi(m)            if m is even OR m==1
        #   A(m) = (3/2)*phi(m)        if m is odd and >= 3
        ph = phi[m]
        if m == 1 or (m & 1) == 0:
            A = (2 * ph) % MOD
        else:
            A = (3 * ph) % MOD
            A = (A * inv2) % MOD

        # G(m) = sum_{g=1..L} 2^{g*(m-1)}  (geometric series)
        if t == 0:
            G = L % MOD
        else:
            r = pow2[t]
            den_inv = inv_den[t]
            if den_inv == 0:
                # ratio r == 1 (mod MOD)
                G = L % MOD
            else:
                rL = pow2[t * L]
                num = (r * (rL - 1)) % MOD
                G = (num * den_inv) % MOD

        ans = (ans + A * G) % MOD

    return ans


def _F_small_exact(n: int, k: int) -> int:
    """
    Exact F(n,k) for small n,k, using the derived closed-form rank:
      deg_gcd = gcd(n,k) - 1 if v2(k) <= v2(n), else gcd(n,k)
      rank = n - deg_gcd
      F = 2^rank
    """
    g = gcd(n, k)
    lowbit_n = n & -n
    lowbit_k = k & -k
    eps = 1 if lowbit_k <= lowbit_n else 0
    deg_gcd = g - eps
    rank = n - deg_gcd
    return 1 << rank


def _self_test() -> None:
    # Test values given in the problem statement
    assert _F_small_exact(3, 2) == 4
    assert _F_small_exact(8, 3) == 256
    assert _F_small_exact(9, 3) == 128

    assert solve(3) == 22
    assert solve(10) == 10444
    assert solve(1000) == 853_837_042  # already modulo MOD


def main() -> None:
    _self_test()
    N = 10_000_000
    if len(sys.argv) > 1:
        N = int(sys.argv[1])
    print(solve(N))


if __name__ == "__main__":
    main()
