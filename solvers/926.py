#!/usr/bin/env python3
"""Project Euler 926: Total Roundness

Computes R(10_000_000!) modulo 1_000_000_007.

No external libraries are used.
"""

from __future__ import annotations

import sys
from array import array
from bisect import bisect_right
from math import isqrt

MOD = 1_000_000_007


def roundness(n: int, b: int) -> int:
    """Number of trailing zeros of n in base b (i.e., max k with b**k | n)."""
    if b <= 1:
        raise ValueError("base must be > 1")
    if n == 0:
        # Conventionally infinite, but not needed for this problem.
        raise ValueError("roundness is undefined for n=0")
    k = 0
    while n % b == 0:
        n //= b
        k += 1
    return k


def _factor_exponents(n: int) -> list[int]:
    """Prime exponents of n (unordered). For small test values."""
    exps: list[int] = []
    if n < 2:
        return exps

    # Factor 2
    e = 0
    while (n & 1) == 0:
        n >>= 1
        e += 1
    if e:
        exps.append(e)

    d = 3
    while d * d <= n:
        e = 0
        while n % d == 0:
            n //= d
            e += 1
        if e:
            exps.append(e)
        d += 2

    if n > 1:
        exps.append(1)
    return exps


def total_roundness(n: int) -> int:
    """Compute R(n) exactly for small n (used for asserts)."""
    exps = _factor_exponents(n)
    if not exps:
        return 0

    emax = max(exps)
    d = [1] * (emax + 1)  # d[k] = number of k-th power divisors (including 1)

    for e in exps:
        for k in range(1, e + 1):
            d[k] *= (e // k) + 1

    return sum(d[1:]) - emax


def _primes_upto(n: int) -> list[int]:
    """List primes <= n using an odd-only sieve."""
    if n < 2:
        return []
    if n == 2:
        return [2]

    size = n // 2 + 1  # index i represents odd number 2*i+1
    sieve = bytearray(b"\x01") * size
    sieve[0] = 0  # 1 is not prime

    limit = isqrt(n)
    half_limit = limit // 2
    for i in range(1, half_limit + 1):
        if sieve[i]:
            p = 2 * i + 1
            start = (p * p) // 2
            sieve[start::p] = b"\x00" * (((size - start - 1) // p) + 1)

    primes = [2]
    primes.extend(2 * i + 1 for i in range(1, size) if sieve[i])
    if primes and primes[-1] > n:
        primes.pop()
    return primes


def _v_p_factorial(n: int, p: int) -> int:
    """Legendre's formula for exponent of prime p in n!."""
    e = 0
    while n:
        n //= p
        e += n
    return e


def _apply_exponent_update(D: array, e: int, power: int, mod: int) -> None:
    """Multiply D[k] by (floor(e/k)+1)^power for all 1<=k<=e.

    Uses quotient grouping so the expensive division is only done per block.
    """
    A = D
    k = 1
    if power == 1:
        while k <= e:
            q = e // k
            r = e // q
            mult = q + 1
            for i in range(k, r + 1):
                A[i] = (A[i] * mult) % mod
            k = r + 1
    else:
        while k <= e:
            q = e // k
            r = e // q
            mult = pow(q + 1, power, mod)
            for i in range(k, r + 1):
                A[i] = (A[i] * mult) % mod
            k = r + 1


def total_roundness_factorial(n: int, mod: int = MOD) -> int:
    """Compute R(n!) modulo mod."""
    if n <= 1:
        return 0

    # Maximum exponent occurs at p=2.
    emax = _v_p_factorial(n, 2)

    # D[k] will become the number of k-th power divisors of n! (including 1), modulo mod.
    D = array("I", [1]) * (emax + 1)

    primes = _primes_upto(n)
    sq = isqrt(n)
    split = bisect_right(primes, sq)

    # Small primes (p <= sqrt(n)) need full Legendre exponent.
    for p in primes[:split]:
        e = _v_p_factorial(n, p)
        _apply_exponent_update(D, e, 1, mod)

    # Large primes (p > sqrt(n)) have exponent floor(n/p); group them by that exponent.
    counts = [0] * (sq + 1)  # exponents are in [1..sq]
    for p in primes[split:]:
        counts[n // p] += 1

    for e in range(1, sq + 1):
        c = counts[e]
        if c:
            _apply_exponent_update(D, e, c, mod)

    # R(n!) = sum_{k>=1} (D[k] - 1). Here k runs to emax; for k>emax the term is 0.
    total = 0
    # D is an array('I'); summing a slice materializes Python ints but is fine.
    total = sum(D[1:]) % mod
    return (total - emax) % mod


def _self_test() -> None:
    # From the statement examples.
    assert roundness(20, 2) == 2
    assert roundness(20, 4) == 1
    assert roundness(20, 5) == 1
    assert roundness(20, 10) == 1
    assert roundness(20, 20) == 1
    assert total_roundness(20) == 6

    # Also given in the statement.
    assert total_roundness_factorial(10) == 312  # exact fits comfortably


def main(argv: list[str]) -> None:
    _self_test()

    n = 10_000_000
    if len(argv) >= 2:
        n = int(argv[1])
        if n < 0:
            raise SystemExit("n must be non-negative")

    print(total_roundness_factorial(n, mod=MOD))


if __name__ == "__main__":
    main(sys.argv)
