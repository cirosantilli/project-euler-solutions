#!/usr/bin/env python3
"""
Project Euler 241: Perfection Quotients

Find the sum of all n <= 10^18 such that sigma(n)/n is a half-integer.

Equivalently:
    2*sigma(n)/n is an odd integer.

This version does NOT hardcode the OEIS list; it generates all solutions
<= limit at runtime via a constrained DFS + fast integer factorization.
"""

from __future__ import annotations

from math import gcd, isqrt
from fractions import Fraction
from functools import lru_cache


# ---------- Small helper: sigma by trial division (only for asserts) ----------


def sigma_trial(n: int) -> int:
    """Sum of divisors σ(n) by trial division (only used for small asserts)."""
    if n <= 1:
        return 1 if n == 1 else 0
    result = 1
    x = n
    p = 2
    while p * p <= x:
        if x % p == 0:
            term = 1
            pp = 1
            while x % p == 0:
                x //= p
                pp *= p
                term += pp
            result *= term
        p += 1 if p == 2 else 2
    if x > 1:
        result *= 1 + x
    return result


# ---------- Prime sieve (for bounding) ----------


def primes_upto(n: int) -> list[int]:
    sieve = bytearray(b"\x01") * (n + 1)
    sieve[:2] = b"\x00\x00"
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            step = i
            start = i * i
            sieve[start : n + 1 : step] = b"\x00" * (((n - start) // step) + 1)
    return [i for i in range(n + 1) if sieve[i]]


PRIMES = primes_upto(5000)


# ---------- Deterministic Miller–Rabin for 64-bit ----------

_MR_BASES_64 = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)


def is_probable_prime(n: int) -> bool:
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small:
        if n == p:
            return True
        if n % p == 0:
            return False

    d = n - 1
    s = 0
    while d % 2 == 0:
        s += 1
        d //= 2

    def check(a: int) -> bool:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            return True
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                return True
        return False

    for a in _MR_BASES_64:
        if a % n == 0:
            continue
        if not check(a):
            return False
    return True


# ---------- Pollard Rho factorization (64-bit safe) ----------


def pollard_rho(n: int) -> int:
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3

    # deterministic-ish cycle parameters
    for c in (1, 3, 5, 7, 11, 13, 17, 19, 23):
        f = lambda x: (x * x + c) % n
        x = 2
        y = 2
        d = 1
        while d == 1:
            x = f(x)
            y = f(f(y))
            d = gcd(abs(x - y), n)
        if d != n:
            return d

    # fallback (should basically never happen here)
    return 1


def factor_int(n: int, out: dict[int, int] | None = None) -> dict[int, int]:
    if out is None:
        out = {}
    if n == 1:
        return out
    if is_probable_prime(n):
        out[n] = out.get(n, 0) + 1
        return out

    d = pollard_rho(n)
    if d == 1 or d == n:
        # should not happen often; treat as prime as last resort
        out[n] = out.get(n, 0) + 1
        return out

    factor_int(d, out)
    factor_int(n // d, out)
    return out


@lru_cache(maxsize=None)
def prime_factors_tuple(n: int) -> tuple[int, ...]:
    return tuple(sorted(factor_int(n).keys()))


# ---------- Arithmetic helpers ----------


def sigma_prime_power(p: int, e: int) -> int:
    # σ(p^e) = (p^(e+1)-1)/(p-1)
    return (pow(p, e + 1) - 1) // (p - 1)


def compute_m_max(limit: int) -> int:
    """
    Upper-bound m = 2σ(n)/n using:
        σ(n)/n <= ∏_{p|n} p/(p-1)
    Max occurs by taking smallest primes until product exceeds limit.
    """
    prod = 1
    I = Fraction(1, 1)
    for p in PRIMES:
        if prod * p > limit:
            break
        prod *= p
        I *= Fraction(p, p - 1)
    mmax = (2 * I.numerator) // I.denominator
    if mmax % 2 == 0:
        mmax -= 1
    return mmax


def max_upper_multiplier(
    n_current: int, used_primes: frozenset[int], limit: int
) -> Fraction:
    """
    Greedy over smallest unused primes, multiplying an upper bound factor p/(p-1),
    while keeping n_current * (product of chosen primes) <= limit.
    """
    mult = Fraction(1, 1)
    prod = 1
    for p in PRIMES:
        if p == 2 or p in used_primes:
            continue
        if n_current * prod * p > limit:
            break
        prod *= p
        mult *= Fraction(p, p - 1)
    return mult


def search_for_m(m: int, limit: int) -> set[int]:
    """
    Find all n <= limit such that 2σ(n)/n == m (m odd),
    via DFS from n = 2^a and only extending by primes that appear
    in the *current reduced numerator*.
    """
    sols: set[int] = set()
    max_a = limit.bit_length() - 1

    for a in range(1, max_a + 1):
        n0 = 1 << a
        if n0 > limit:
            break

        # Start from n=2^a:
        # 2σ(2^a)/2^a = (2^(a+1)-1)/2^(a-1)
        num0 = (1 << (a + 1)) - 1
        den0 = 1 << (a - 1)
        g = gcd(num0, den0)
        num0 //= g
        den0 //= g

        stack = [(n0, num0, den0, frozenset([2]))]
        seen_n: set[int] = set()

        while stack:
            n, num, den, used = stack.pop()

            if n in seen_n:
                continue
            seen_n.add(n)

            if num > m * den:
                continue

            # if even with best-case extra primes we can't reach m, prune
            q = Fraction(num, den)
            if q * max_upper_multiplier(n, used, limit) < m:
                continue

            if den == 1 and num == m:
                sols.add(n)
                continue

            for p in prime_factors_tuple(num):
                if p == 2 or p in used:
                    continue

                pe = p
                e = 1
                while n * pe <= limit:
                    sig = sigma_prime_power(p, e)
                    new_num = num * sig
                    new_den = den * pe
                    gg = gcd(new_num, new_den)
                    new_num //= gg
                    new_den //= gg

                    if new_num > m * new_den:
                        break

                    stack.append((n * pe, new_num, new_den, used | {p}))

                    e += 1
                    pe *= p

    return sols


# ---------- Public solve() ----------


def solve(limit: int = 10**18) -> int:
    mmax = compute_m_max(limit)
    sols: set[int] = set()
    for m in range(3, mmax + 1, 2):
        sols |= search_for_m(m, limit)
    return sum(sols)


def main() -> None:
    # Project Euler statement example:
    assert sigma_trial(6) == 12, "Problem statement example: σ(6) must equal 12"
    print(solve(10**18))


if __name__ == "__main__":
    main()
