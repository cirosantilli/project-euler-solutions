#!/usr/bin/env python3
"""
Project Euler 797: Cyclogenic Polynomials

Compute Q_{10^7}(2) modulo 1_000_000_007.

No external libraries are used (only Python standard library).
"""
from array import array
import sys


MOD = 1_000_000_007


def linear_sieve_mu_spf(n: int):
    """
    Linear sieve that returns:
      - mu[0..n]  (Mobius function, values in {-1,0,1})
      - spf[0..n] (smallest prime factor; spf[1]=1)
    """
    spf = array("I", [0]) * (n + 1)
    mu = array("b", [0]) * (n + 1)
    primes = []

    spf[1] = 1
    mu[1] = 1

    for i in range(2, n + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
            mu[i] = -1
        for p in primes:
            ip = i * p
            if ip > n:
                break
            spf[ip] = p
            if i % p == 0:
                mu[ip] = 0
                break
            mu[ip] = -mu[i]

    return mu, spf


def build_2pow_minus1_and_inverses(n: int, mod: int):
    """
    Build arrays:
      b[k]   = 2^k - 1 (mod mod), for k=0..n (b[0]=0)
      invb[k]= (2^k - 1)^(-1) (mod mod), for k=1..n, computed with one exponentiation
               invb[0]=0 (unused)
    Assumption: for k<=n, (2^k - 1) != 0 (mod mod). This holds here since n=10^7 and
    the multiplicative order of 2 mod 1_000_000_007 is much larger than 10^7.
    """
    b = array("I", [0]) * (n + 1)
    invb = array("I", [0]) * (n + 1)

    pow2 = 1
    for k in range(1, n + 1):
        pow2 = (pow2 * 2) % mod
        b[k] = pow2 - 1  # always non-negative in Python ints

    # Batch inversion: store prefix products in invb first.
    invb[0] = 1
    acc = 1
    for k in range(1, n + 1):
        acc = (acc * b[k]) % mod
        invb[k] = acc

    inv_total = pow(acc, mod - 2, mod)
    for k in range(n, 0, -1):
        prev_prefix = invb[k - 1]
        invb[k] = (inv_total * prev_prefix) % mod
        inv_total = (inv_total * b[k]) % mod

    invb[0] = 0
    return b, invb


def cyclotomic_value_at_2(n: int, spf, b, invb, mod: int) -> int:
    """
    Compute Φ_n(2) (mod mod) using:
        Φ_n(2) = ∏_{s | rad(n)} (2^{n/s} - 1)^{μ(s)}
    where μ(s)=(-1)^{ω(s)} for squarefree s.

    The key point: only squarefree divisors s of rad(n) appear, so there are 2^{ω(n)} terms.
    """
    if n == 1:
        return 1

    # Extract distinct primes of n
    m = n
    primes = []
    while m > 1:
        p = spf[m]
        primes.append(p)
        while m % p == 0:
            m //= p

    # Enumerate squarefree divisors of rad(n) and their parity (even/odd subset size)
    prods = [1]
    parity = [0]  # 0 -> even, 1 -> odd
    for p in primes:
        L = len(prods)
        for i in range(L):
            prods.append(prods[i] * p)
            parity.append(parity[i] ^ 1)

    res = 1
    for s, par in zip(prods, parity):
        idx = n // s  # n/s
        if par == 0:
            res = (res * b[idx]) % mod
        else:
            res = (res * invb[idx]) % mod
    return res


def build_T_prefix(n: int, spf, b, invb, mod: int):
    """
    Build T[k] = ∏_{d|k} (1 + Φ_d(2)) (mod mod), then convert in-place to prefix sums:
      T[k] <- ∑_{i=1..k} T[i] (mod mod)

    Returns the prefix array (same object).
    """
    T = array("I", [1]) * (n + 1)
    T[0] = 0
    N1 = n + 1
    T_local = T
    mod_local = mod

    for d in range(1, N1):
        phi_d = cyclotomic_value_at_2(d, spf, b, invb, mod_local)
        fd = phi_d + 1
        if fd >= mod_local:
            fd -= mod_local

        for m in range(d, N1, d):
            T_local[m] = (T_local[m] * fd) % mod_local

    # Prefix sum in-place
    run = 0
    for i in range(1, N1):
        run += T_local[i]
        if run >= mod_local:
            run -= mod_local
        T_local[i] = run

    return T_local


def solve(n: int = 10_000_000) -> int:
    """
    Returns Q_n(2) (mod MOD).
    """
    mu, spf = linear_sieve_mu_spf(n)
    b, invb = build_2pow_minus1_and_inverses(n, MOD)

    # Build prefix sums of T_k
    T_prefix = build_T_prefix(n, spf, b, invb, MOD)

    # Release big arrays we no longer need ASAP (helps peak memory)
    del b, invb, spf

    # Prefix sums of mu (Mertens function)
    prefix_mu = array("i", [0]) * (n + 1)
    run = 0
    for i in range(1, n + 1):
        run += mu[i]
        prefix_mu[i] = run

    # Q_n = sum_{d<=n} mu(d) * (sum_{k<=n/d} T_k)
    # Group by constant t = floor(n/d)
    ans = 0
    l = 1
    while l <= n:
        t = n // l
        r = n // t
        sum_mu = prefix_mu[r] - prefix_mu[l - 1]
        ans = (ans + (sum_mu % MOD) * T_prefix[t]) % MOD
        l = r + 1

    return ans


def _divisors_small(x: int):
    ds = []
    d = 1
    while d * d <= x:
        if x % d == 0:
            ds.append(d)
            if d * d != x:
                ds.append(x // d)
        d += 1
    return ds


def _example_asserts():
    """
    Asserts for the examples given in the problem statement:
      - P_6(2) = 234 (from the provided P_6(x))
      - Q_10(2) = 5598
    These are computed from scratch with a small limit, so they're fast.
    """
    N = 10
    mu, spf = linear_sieve_mu_spf(N)
    b, invb = build_2pow_minus1_and_inverses(N, MOD)

    # Build raw T (not prefix) for N=10
    T = array("I", [1]) * (N + 1)
    T[0] = 0
    for d in range(1, N + 1):
        phi_d = cyclotomic_value_at_2(d, spf, b, invb, MOD)
        fd = phi_d + 1
        if fd >= MOD:
            fd -= MOD
        for m in range(d, N + 1, d):
            T[m] = (T[m] * fd) % MOD

    def P_of(n: int) -> int:
        s = 0
        for d in _divisors_small(n):
            s += mu[d] * T[n // d]
        return s % MOD

    assert P_of(6) == 234
    q10 = 0
    for k in range(1, 11):
        q10 = (q10 + P_of(k)) % MOD
    assert q10 == 5598


def main():
    _example_asserts()

    n = 10_000_000
    if len(sys.argv) >= 2:
        n = int(sys.argv[1])

    print(solve(n))


if __name__ == "__main__":
    main()
