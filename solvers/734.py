#!/usr/bin/env python3
"""
Project Euler 734 - A Bit of Prime

Compute T(n, k): the number of k-tuples of primes <= n whose bitwise-OR is a prime <= n,
modulo 1_000_000_007.

No external libraries are used.
"""

from math import isqrt

MOD = 1_000_000_007


def sieve_primes_upto(n: int) -> bytearray:
    """Return bytearray is_prime[0..n] with 1 for primes, 0 otherwise."""
    if n < 1:
        return bytearray(b"\x00") * (n + 1)
    is_prime = bytearray(b"\x01") * (n + 1)
    is_prime[0:2] = b"\x00\x00"
    limit = isqrt(n)
    for p in range(2, limit + 1):
        if is_prime[p]:
            start = p * p
            step = p
            # Mark multiples with slice assignment (fast in CPython)
            is_prime[start : n + 1 : step] = b"\x00" * (((n - start) // step) + 1)
    return is_prime


def T(n: int, k: int, mod: int = MOD) -> int:
    """
    Compute T(n,k) mod mod using zeta transforms on bitmasks.

    The bit-length of n determines the mask width B, and we work in [0, 2^B).
    """
    B = n.bit_length()
    size = 1 << B

    is_prime = sieve_primes_upto(n)

    # a[m] = number of primes x <= n with x being a submask of m (x & ~m == 0)
    # s[m] will become sum_{prime y superset m} (-1)^{popcount(y)}
    a = [0] * size
    s = [0] * size

    for p in range(2, n + 1):
        if is_prime[p]:
            a[p] = 1
            s[p] = -1 if (p.bit_count() & 1) else 1

    # Subset zeta transform: a[m] = sum_{x subset m} f[x]
    for i in range(B):
        step = 1 << i
        jump = step << 1
        for base in range(0, size, jump):
            lo = base + step
            hi = base + jump
            # a[lo:hi] += a[base:base+step] elementwise
            for m in range(lo, hi):
                a[m] += a[m - step]

    # Superset zeta transform: s[m] = sum_{y superset m} h[y]
    for i in range(B):
        step = 1 << i
        jump = step << 1
        for base in range(0, size, jump):
            mid = base + step
            for m in range(base, mid):
                s[m] += s[m + step]

    # After swapping inclusion-exclusion sums:
    # T = sum_m a[m]^k * (-1)^{popcount(m)} * s[m]
    res = 0
    for m in range(n + 1):
        base = a[m]
        if base == 0:
            continue
        val = pow(base, k, mod)
        coeff = s[m]
        if m.bit_count() & 1:
            coeff = -coeff
        res = (res + val * coeff) % mod
    return res


def main() -> None:
    # Test values from the problem statement
    assert T(5, 2) == 5
    assert T(100, 3) == 3355
    assert T(1000, 10) == 2071632

    n = 10**6
    k = 999_983
    print(T(n, k))


if __name__ == "__main__":
    main()
