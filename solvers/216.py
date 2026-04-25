#!/usr/bin/env python
"""
Project Euler 216: Investigating the primality of 2n^2 - 1

Sieve the indices n for which some prime p divides 2n^2 - 1.  Survivors are
exactly the prime values.
"""

from math import isqrt


LIMIT = 50_000_000


def sieve_primes(limit: int) -> list[int]:
    if limit < 2:
        return []
    flags = bytearray(b"\x01") * (limit + 1)
    flags[0:2] = b"\x00\x00"
    for p in range(2, isqrt(limit) + 1):
        if flags[p]:
            start = p * p
            flags[start : limit + 1 : p] = b"\x00" * (((limit - start) // p) + 1)
    return [p for p in range(2, limit + 1) if flags[p]]


def tonelli_shanks(a: int, p: int) -> int:
    if a == 0:
        return 0
    if pow(a, (p - 1) // 2, p) != 1:
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


def mark_progression(composite: bytearray, limit: int, start: int, step: int) -> None:
    if start < 2:
        start += ((2 - start + step - 1) // step) * step
    if start > limit:
        return
    count = (limit - start) // step + 1
    composite[start : limit + 1 : step] = b"\x01" * count


def solve(limit: int = LIMIT) -> int:
    if limit < 2:
        return 0

    max_value = 2 * limit * limit - 1
    primes = sieve_primes(isqrt(max_value))
    composite = bytearray(limit + 1)

    for p in primes:
        if p == 2:
            continue
        if p & 7 not in (1, 7):
            continue

        inv2 = (p + 1) // 2
        if p & 7 == 7:
            root = pow(2, (p + 1) // 4, p) * inv2 % p
        else:
            root = tonelli_shanks(inv2, p)
            if root == 0:
                continue

        mark_progression(composite, limit, root, p)
        other = (-root) % p
        if other != root:
            mark_progression(composite, limit, other, p)

        maybe_self = isqrt(inv2)
        if maybe_self * maybe_self == inv2 and maybe_self <= limit:
            composite[maybe_self] = 0

    return composite[2:].count(0)


def main() -> None:
    assert solve(10_000) == 2202
    print(solve())


if __name__ == "__main__":
    main()
