#!/usr/bin/env python3
"""
Project Euler 801: x^y ≡ y^x (mod p)

We need S(10^16, 10^16 + 10^6) modulo 993353399, where
S(M,N) = sum_{primes p in [M,N]} f(p),
and f(n) counts pairs 0 < x,y <= n^2-n such that x^y ≡ y^x (mod n).

This implementation uses:
- A number-theoretic reduction for prime modulus p
- Multiplicativity on p-1 via CRT
- Deterministic Miller-Rabin for 64-bit primality
- Pollard-Rho factorization for 64-bit integers
- Segmented sieve (small primes) to reduce primality checks in [10^16, 10^16+10^6]
"""

from __future__ import annotations

import math
from typing import Dict, List, Iterator, Tuple

MOD = 993353399

# -----------------------------
# Small prime sieve
# -----------------------------


def sieve_primes(n: int) -> List[int]:
    """Return list of all primes <= n (n up to a few million is fine)."""
    if n < 2:
        return []
    bs = bytearray(b"\x01") * (n + 1)
    bs[0:2] = b"\x00\x00"
    r = int(math.isqrt(n))
    for p in range(2, r + 1):
        if bs[p]:
            step = p
            start = p * p
            bs[start : n + 1 : step] = b"\x00" * (((n - start) // step) + 1)
    return [i for i in range(2, n + 1) if bs[i]]


# -----------------------------
# 64-bit deterministic Miller-Rabin
# -----------------------------

_MR_BASES_64 = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)
_SMALL_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def is_prime_64(n: int) -> bool:
    """Deterministic primality test for 0 <= n < 2^64."""
    if n < 2:
        return False
    for p in _SMALL_PRIMES:
        if n == p:
            return True
        if n % p == 0:
            return False

    # write n-1 = d * 2^s with d odd
    d = n - 1
    s = 0
    while d % 2 == 0:
        s += 1
        d //= 2

    for a in _MR_BASES_64:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        composite = True
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                composite = False
                break
        if composite:
            return False
    return True


# -----------------------------
# Pollard-Rho factorization (64-bit)
# -----------------------------

_rng_state = 0x9E3779B97F4A7C15  # fixed seed for reproducibility


def _rand64() -> int:
    global _rng_state
    # xorshift64*
    x = _rng_state & ((1 << 64) - 1)
    x ^= (x >> 12) & ((1 << 64) - 1)
    x ^= (x << 25) & ((1 << 64) - 1)
    x ^= (x >> 27) & ((1 << 64) - 1)
    _rng_state = x
    return (x * 2685821657736338717) & ((1 << 64) - 1)


def _pollard_rho(n: int) -> int:
    """Return a non-trivial factor of composite odd n."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3

    while True:
        c = (_rand64() % (n - 1)) + 1
        x = (_rand64() % (n - 2)) + 2
        y = x
        d = 1

        # f(z) = z^2 + c mod n
        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = math.gcd(abs(x - y), n)

        if d != n:
            return d


def factorize_64(n: int) -> Dict[int, int]:
    """Prime factorization of n (n fits in signed 64-bit), returned as {prime: exponent}."""
    factors: Dict[int, int] = {}
    if n <= 1:
        return factors

    # Strip small primes quickly (up to 10k)
    # This cuts down the work for Pollard-Rho significantly for typical inputs.
    for p in _TRIAL_PRIMES_10K:
        if p * p > n:
            break
        if n % p == 0:
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            factors[p] = factors.get(p, 0) + e
    if n == 1:
        return factors
    if is_prime_64(n):
        factors[n] = factors.get(n, 0) + 1
        return factors

    stack = [n]
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        if is_prime_64(m):
            factors[m] = factors.get(m, 0) + 1
            continue
        d = _pollard_rho(m)
        stack.append(d)
        stack.append(m // d)
    return factors


# Precompute trial division primes up to 10k once.
_TRIAL_PRIMES_10K = sieve_primes(10_000)

# -----------------------------
# Core counting: f(p) for prime p
# -----------------------------


def _g_prime_power(q: int, e: int, mod: int | None) -> int:
    """
    g(q^e): number of solutions (u,A,v,B) mod q^e with uA ≡ vB (mod q^e).

    Closed form (derived from divisor-structure for prime powers):
      Let m = q^e.
      g(m) = (q-1)^3 * sum_{t=1..e} t^2 * q^{3e-t-2}  +  q^{2e-2} * (e(q-1)+q)^2
    """
    if e <= 0:
        raise ValueError("Exponent must be positive")

    if mod is None:
        s = 0
        for t in range(1, e + 1):
            s += t * t * (q ** (3 * e - t - 2))
        return (q - 1) ** 3 * s + (q ** (2 * e - 2)) * (e * (q - 1) + q) ** 2

    qm = q % mod
    s = 0
    for t in range(1, e + 1):
        exp = 3 * e - t - 2
        s = (s + (t * t % mod) * pow(qm, exp, mod)) % mod

    term1 = pow((q - 1) % mod, 3, mod) * s % mod
    term2 = pow(qm, 2 * e - 2, mod) * ((e * (q - 1) + q) % mod) ** 2 % mod
    return (term1 + term2) % mod


def _g_from_factorization(factors: Dict[int, int], mod: int | None) -> int:
    """g(m) multiplicatively from m's prime-power factorization."""
    g_val = 1
    for q, e in factors.items():
        g_val = (
            g_val * _g_prime_power(q, e, mod)
            if mod is None
            else (g_val * _g_prime_power(q, e, mod)) % mod
        )
    return g_val


def f_of_prime(p: int, mod: int | None = None) -> int:
    """
    Compute f(p) for prime p.

    For prime p:
      f(p) = (p-1)^2  +  g(p-1),
    where g(m) counts solutions (u,A,v,B) in (Z_m)^4 to uA ≡ vB (mod m).
    """
    m = p - 1
    factors = factorize_64(m)

    if mod is None:
        g_val = _g_from_factorization(factors, None)
        return m * m + g_val

    g_val = _g_from_factorization(factors, mod)
    mm = m % mod
    return (mm * mm + g_val) % mod


# -----------------------------
# Prime generation in a large interval
# -----------------------------


def primes_in_interval(
    lo: int, hi: int, pre_sieve_limit: int = 200_000
) -> Iterator[int]:
    """
    Generate primes in [lo, hi] by:
      1) segmented marking by primes <= pre_sieve_limit
      2) deterministic Miller-Rabin to confirm primality
    """
    if hi < lo:
        return
    length = hi - lo + 1
    is_comp = bytearray(length)  # 0 = maybe prime, 1 = composite (by small primes)

    small = sieve_primes(pre_sieve_limit)
    for q in small:
        # mark all indices i with lo+i ≡ 0 mod q
        offset = (-lo) % q
        for i in range(offset, length, q):
            is_comp[i] = 1
        # don't mark the prime itself if it lies inside [lo, hi]
        if lo <= q <= hi:
            is_comp[q - lo] = 0

    # handle 0 and 1 if they are inside range
    for v in (0, 1):
        if lo <= v <= hi:
            is_comp[v - lo] = 1

    for i in range(length):
        if is_comp[i]:
            continue
        n = lo + i
        if is_prime_64(n):
            yield n


# -----------------------------
# Summation S(M,N)
# -----------------------------


def S(M: int, N: int, mod: int) -> int:
    """Sum f(p) over primes p in [M,N], result modulo mod."""
    total = 0
    for p in primes_in_interval(M, N):
        total = (total + f_of_prime(p, mod)) % mod
    return total


# -----------------------------
# Self-tests from the problem statement
# -----------------------------


def _self_test() -> None:
    # f(5)=104 and f(97)=1614336
    assert f_of_prime(5, None) == 104
    assert f_of_prime(97, None) == 1614336

    # S(1, 10^2)=7381000
    # For small ranges, use a direct prime sieve for convenience and speed.
    primes_100 = sieve_primes(100)
    s100 = 0
    for p in primes_100:
        s100 += f_of_prime(p, None)
    assert s100 == 7381000

    # S(1, 10^5) ≡ 701331986 (mod 993353399)
    primes_1e5 = sieve_primes(100_000)
    s1e5 = 0
    for p in primes_1e5:
        s1e5 = (s1e5 + f_of_prime(p, MOD)) % MOD
    assert s1e5 == 701331986


# -----------------------------
# Entry point
# -----------------------------


def main() -> None:
    _self_test()
    lo = 10**16
    hi = lo + 10**6
    print(S(lo, hi, MOD))


if __name__ == "__main__":
    main()
