#!/usr/bin/env python
"""
Project Euler 241: Perfection Quotients

Find the sum of all n <= 10^18 such that sigma(n)/n is a half-integer.

Equivalently:
    2*sigma(n)/n is an odd integer.

This version does NOT hardcode the OEIS list; it generates all solutions
<= limit at runtime via a constrained DFS + fast integer factorization.
"""

from __future__ import annotations

from math import gcd
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
def factor_items_tuple(n: int) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(factor_int(n).items()))


# ---------- Arithmetic helpers ----------


def sigma_prime_power(p: int, e: int) -> int:
    # σ(p^e) = (p^(e+1)-1)/(p-1)
    return (pow(p, e + 1) - 1) // (p - 1)


def search_target(target_numerator: int, limit: int) -> list[int]:
    """
    Search for sigma(n) / n == target_numerator / 2.

    A state stores Q = T*n/sigma(n) = numerator/denominator.  In any completion,
    the denominator must divide the remaining cofactor, so its smallest prime
    divisor is the next forced prime.
    """
    solutions: list[int] = []
    stack = [(1, target_numerator, 2, ())]

    while stack:
        n, numerator, denominator, used_primes = stack.pop()

        if numerator == denominator:
            solutions.append(n)
            continue
        if numerator < denominator:
            continue
        if n * denominator > limit:
            continue
        if denominator == 1:
            continue

        p, min_exponent = factor_items_tuple(denominator)[0]
        if p in used_primes:
            continue

        prime_power = 1
        sigma_power = 1
        for _ in range(min_exponent):
            prime_power *= p
            sigma_power += prime_power

        while n * prime_power <= limit:
            new_numerator = numerator * prime_power
            new_denominator = denominator * sigma_power
            common = gcd(new_numerator, new_denominator)
            new_numerator //= common
            new_denominator //= common

            if new_numerator < new_denominator:
                break

            stack.append(
                (
                    n * prime_power,
                    new_numerator,
                    new_denominator,
                    used_primes + (p,),
                )
            )

            prime_power *= p
            sigma_power += prime_power

    return solutions


# ---------- Public solve() ----------


def solve(limit: int = 10**18) -> int:
    return sum(
        sum(search_target(target_numerator, limit))
        for target_numerator in (3, 5, 7, 9, 11, 13)
    )


def main() -> None:
    # Project Euler statement example:
    assert sigma_trial(6) == 12, "Problem statement example: σ(6) must equal 12"
    print(solve(10**18))


if __name__ == "__main__":
    main()
