#!/usr/bin/env python3
"""
Project Euler 805: Shifted Multiples

Compute T(200) modulo 1_000_000_007.

No external libraries are used. The program includes asserts for all example
values stated in the problem statement.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

MOD = 1_000_000_007
M = 200  # problem asks for T(200)


def primes_upto(limit: int) -> list[int]:
    """Simple sieve of Eratosthenes up to `limit` (inclusive)."""
    if limit < 2:
        return []
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[0:2] = b"\x00\x00"
    r = int(limit**0.5)
    for p in range(2, r + 1):
        if sieve[p]:
            step = p
            start = p * p
            sieve[start : limit + 1 : step] = b"\x00" * (((limit - start) // step) + 1)
    return [i for i in range(limit + 1) if sieve[i]]


# For M=200 we have b=v^3 <= 8,000,000 and D=10b-a <= 80,000,000.
# Factoring numbers up to ~80 million only needs primes up to sqrt(80e6) < 9000.
_PRIMES = primes_upto(10_000)


def factorize(n: int) -> Dict[int, int]:
    """Prime factorization of n>=1 as {prime: exponent}."""
    res: Dict[int, int] = {}
    x = n
    for p in _PRIMES:
        if p * p > x:
            break
        if x % p == 0:
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            res[p] = e
    if x > 1:
        res[x] = res.get(x, 0) + 1
    return res


def euler_phi(n: int) -> int:
    """Euler's totient function for n>=1."""
    if n == 1:
        return 1
    fac = factorize(n)
    phi = n
    for p in fac.keys():
        phi = (phi // p) * (p - 1)
    return phi


_order_cache: Dict[int, Optional[int]] = {}


def multiplicative_order_10(m: int) -> Optional[int]:
    """
    Return the multiplicative order of 10 modulo m, i.e. the smallest k>0
    with 10^k ≡ 1 (mod m). If gcd(10,m) != 1, return None.
    """
    if m == 1:
        return 1
    cached = _order_cache.get(m)
    if cached is not None:
        return cached
    if math.gcd(m, 10) != 1:
        _order_cache[m] = None
        return None

    phi = euler_phi(m)
    fac_phi = factorize(phi)

    k = phi
    for p in fac_phi.keys():
        while k % p == 0 and pow(10, k // p, m) == 1:
            k //= p

    _order_cache[m] = k
    return k


def digit_k_bounds(a: int, b: int, d: int) -> Tuple[Optional[int], Optional[int]]:
    """
    Derive bounds on the digit length k for solutions with leading digit d.

    Let r = a/b (in lowest terms), D = 10b - a (>0).

    From s(n) = r*n and writing n as d followed by (k-1) digits, we get:
        n = d*b*(10^k - 1) / D   (must be an integer)
    and the leading-digit constraint becomes two simple inequalities in 10^(k-1):

    Lower bound (ensures the leading digit is at least d):
        a * 10^(k-1) >= b

    Upper bound is only active if a(d+1) > 10b:
        10^(k-1) < b*d / (a(d+1) - 10b)

    Returns (k_low, k_high), where k_high=None means unbounded above.
    If no k is possible, returns (None, None).
    """
    # Lower: find minimal e=k-1 with a*10^e >= b
    e = 0
    t = a
    while t < b:
        t *= 10
        e += 1
    k_low = e + 1

    denom = a * (d + 1) - 10 * b
    if denom <= 0:
        return k_low, None

    # Strict: 10^(k-1) < b*d/denom  =>  10^(k-1) <= floor((b*d - 1)/denom)
    max_pow10 = (b * d - 1) // denom
    if max_pow10 <= 0:
        return None, None

    # Find largest e_high with 10^e_high <= max_pow10
    e_high = 0
    pow10 = 1
    while pow10 * 10 <= max_pow10:
        pow10 *= 10
        e_high += 1

    k_high = e_high + 1
    return k_low, k_high


def find_N_params(a: int, b: int) -> Optional[Tuple[int, int, int]]:
    """
    For reduced a/b, find minimal (k,d,D) describing N(a/b), or None if no solution.

    D = 10b - a. We also require D>0 (i.e. a/b < 10), otherwise N=0.
    """
    if a == b:
        # Only one-digit numbers satisfy s(n)=n, so N(1)=1.
        return (1, 1, 9 * b)

    D = 10 * b - a
    if D <= 0:
        return None

    best: Optional[Tuple[int, int]] = None  # (k, d)
    for d in range(1, 10):
        k_low, k_high = digit_k_bounds(a, b, d)
        if k_low is None:
            continue
        if k_low < 2:
            k_low = 2  # k=1 implies s(n)=n, only possible for r=1

        # Integer condition: D | d*b*(10^k - 1)
        # Let m = D / gcd(D, d*b). Then we need 10^k ≡ 1 (mod m).
        m = D // math.gcd(D, d * b)
        ord10 = multiplicative_order_10(m)
        if ord10 is None:
            continue

        # Smallest multiple of ord10 that is >= k_low
        k = ((k_low + ord10 - 1) // ord10) * ord10
        if k_high is not None and k > k_high:
            continue

        cand = (k, d)
        if best is None or cand < best:
            best = cand

    if best is None:
        return None
    return (best[0], best[1], D)


def N_mod(a: int, b: int, mod: int = MOD) -> int:
    """Return N(a/b) modulo `mod`."""
    g = math.gcd(a, b)
    a //= g
    b //= g

    if a == b:
        return 1 % mod

    params = find_N_params(a, b)
    if params is None:
        return 0

    k, d, D = params
    invD = pow(D, mod - 2, mod)  # D < mod, so invertible
    return (
        (d % mod) * (b % mod) % mod * ((pow(10, k, mod) - 1) % mod) % mod * invD % mod
    )


def N_exact_small(a: int, b: int, max_k: int = 60) -> int:
    """
    Compute the exact integer N(a/b) when the digit length k is reasonably small.
    Intended for validating the problem statement's examples.
    """
    g = math.gcd(a, b)
    a //= g
    b //= g

    if a == b:
        return 1

    params = find_N_params(a, b)
    if params is None:
        return 0

    k, d, D = params
    if k > max_k:
        raise ValueError("Exact value too large for this validation helper")

    n = d * b * (10**k - 1) // D

    # Validate s(n) = (a/b)*n
    s = int(str(n)[1:] + str(n)[0]) if n >= 10 else n
    assert s * b == n * a
    return n


def T(M_: int, mod: int = MOD) -> int:
    """Compute T(M_) modulo mod."""
    cubes = [0] * (M_ + 1)
    for i in range(1, M_ + 1):
        cubes[i] = i * i * i

    total = 0
    for u in range(1, M_ + 1):
        a = cubes[u]
        for v in range(1, M_ + 1):
            if math.gcd(u, v) != 1:
                continue
            b = cubes[v]
            total = (total + N_mod(a, b, mod)) % mod
    return total


def _self_test() -> None:
    # Examples from the problem statement:
    assert N_exact_small(3, 1) == 142857
    assert N_exact_small(1, 10) == 10
    assert N_mod(2, 1) == 0
    assert T(3) == 262429173


def main() -> None:
    _self_test()
    print(T(M))


if __name__ == "__main__":
    main()
