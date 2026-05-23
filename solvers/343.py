#!/usr/bin/env python
"""
Project Euler 343: Fractional Sequences

The fraction process gives f(n) = LPF(n + 1) - 1.  For cubes,
k^3 + 1 = (k + 1)(k^2 - k + 1), so we sieve largest prime factors of both
factors for all k up to the limit.
"""

from math import isqrt


LIMIT = 2_000_000


def primes_upto(limit: int) -> list[int]:
    if limit < 2:
        return []
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[0:2] = b"\x00\x00"
    for p in range(2, isqrt(limit) + 1):
        if sieve[p]:
            start = p * p
            sieve[start : limit + 1 : p] = b"\x00" * (((limit - start) // p) + 1)
    return [p for p in range(2, limit + 1) if sieve[p]]


def tonelli_shanks(a: int, p: int) -> int:
    """Return one square root of a modulo the odd prime p."""
    a %= p
    if a == 0:
        return 0
    if p % 4 == 3:
        return pow(a, (p + 1) // 4, p)

    q = p - 1
    s = 0
    while q % 2 == 0:
        q //= 2
        s += 1

    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1

    m = s
    c = pow(z, q, p)
    t = pow(a, q, p)
    r = pow(a, (q + 1) // 2, p)

    while t != 1:
        i = 1
        t2 = t * t % p
        while t2 != 1:
            t2 = t2 * t2 % p
            i += 1
        b = pow(c, 1 << (m - i - 1), p)
        r = r * b % p
        t = t * b % p * b % p
        c = b * b % p
        m = i

    return r


def largest_prime_factors_upto(limit: int, primes: list[int]) -> list[int]:
    lpf = [0] * (limit + 1)
    for p in primes:
        for multiple in range(p, limit + 1, p):
            lpf[multiple] = p
    return lpf


def sieve_quadratic_lpf(limit: int, primes: list[int]) -> list[int]:
    rem = [0] * (limit + 1)
    lpf = [1] * (limit + 1)
    for k in range(1, limit + 1):
        rem[k] = k * k - k + 1

    def strip_prime_from_class(p: int, root: int) -> None:
        start = root if root else p
        for k in range(start, limit + 1, p):
            value = rem[k]
            if value % p != 0:
                continue
            while value % p == 0:
                value //= p
            rem[k] = value
            lpf[k] = p

    for p in primes:
        if p == 2:
            continue
        if p == 3:
            strip_prime_from_class(3, 2)
            continue
        if p % 3 != 1:
            continue

        sqrt_disc = tonelli_shanks(p - 3, p)
        inv2 = (p + 1) // 2
        root1 = (1 + sqrt_disc) * inv2 % p
        root2 = (1 - sqrt_disc) * inv2 % p
        strip_prime_from_class(p, root1)
        if root2 != root1:
            strip_prime_from_class(p, root2)

    for k in range(1, limit + 1):
        if rem[k] > 1:
            lpf[k] = rem[k]
    return lpf


def f_simple(n: int) -> int:
    x = 1
    y = n
    while y != 1:
        x += 1
        y -= 1
        g = gcd(x, y)
        x //= g
        y //= g
    return x


def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def solve(limit: int = LIMIT) -> int:
    primes = primes_upto(limit + 1)
    lpf_small = largest_prime_factors_upto(limit + 1, primes)
    lpf_q = sieve_quadratic_lpf(limit, primes)

    total = 0
    for k in range(1, limit + 1):
        total += max(lpf_small[k + 1], lpf_q[k]) - 1
    return total


def main() -> None:
    assert f_simple(1) == 1
    assert f_simple(2) == 2
    assert f_simple(3) == 1
    assert f_simple(20) == 6
    assert solve(100) == 118_937
    print(solve())


if __name__ == "__main__":
    main()
