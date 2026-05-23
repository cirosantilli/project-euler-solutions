#!/usr/bin/env python
"""Project Euler 880: Nested Radicals."""

from __future__ import annotations

import math

N = 10**15
MOD = 1031**3 + 2


def icbrt(n: int) -> int:
    """Floor integer cube root for n >= 0."""
    if n <= 1:
        return n
    r = int(round(n ** (1.0 / 3.0)))
    while (r + 1) ** 3 <= n:
        r += 1
    while r**3 > n:
        r -= 1
    return r


def iroot4(n: int) -> int:
    """Floor integer fourth root for n >= 0."""
    r = math.isqrt(math.isqrt(n))
    while (r + 1) ** 4 <= n:
        r += 1
    while r**4 > n:
        r -= 1
    return r


def cube_free_table(limit: int) -> list[int]:
    """Return cf[n] = product p^(v_p(n) mod 3) for 0 <= n <= limit."""
    spf = list(range(limit + 1))
    if limit >= 1:
        spf[1] = 1

    for p in range(2, math.isqrt(limit) + 1):
        if spf[p] != p:
            continue
        for m in range(p * p, limit + 1, p):
            if spf[m] == m:
                spf[m] = p

    cf = [1] * (limit + 1)
    for n in range(2, limit + 1):
        p = spf[n]
        m = n // p
        e = 1
        while m % p == 0:
            m //= p
            e += 1

        rem = e % 3
        if rem == 0:
            cf[n] = cf[m]
        elif rem == 1:
            cf[n] = cf[m] * p
        else:
            cf[n] = cf[m] * p * p
    return cf


def sumsq(k: int) -> int:
    """1^2 + 2^2 + ... + k^2."""
    return k * (k + 1) * (2 * k + 1) // 6


def H_mod(limit: int) -> int:
    """Compute H(limit) modulo MOD."""
    b_limit = iroot4(4 * limit)

    max_odd_a = max(0, (icbrt(limit) - 1) // 4)
    max_even_a = max(0, (icbrt(limit // 4) - 1) // 2)
    cf_limit = max(4 * max_odd_a, max_even_a, 2 * b_limit)
    cf = cube_free_table(cf_limit)
    cf4 = [0] * (max_odd_a + 1)
    for a in range(1, max_odd_a + 1):
        cf4[a] = cf[4 * a]

    total = 0
    gcd = math.gcd
    isqrt = math.isqrt
    icbrt_local = icbrt
    sumsq_local = sumsq
    mod = MOD

    for b in range(1, b_limit + 1, 2):
        a_limit = (icbrt_local(limit // b) - b) // 4
        cf_b = cf[b]
        for a in range(1, a_limit + 1):
            if gcd(a, b) != 1 or cf_b == cf4[a]:
                continue

            x_base = b + 4 * a
            x = b * x_base * x_base * x_base
            y_base = a - 2 * b
            y_abs = abs(4 * a * y_base * y_base * y_base)
            max_coord = x if x >= y_abs else y_abs
            if max_coord > limit:
                continue

            tmax = isqrt(limit // max_coord)
            total = (total + (x + y_abs) * sumsq_local(tmax)) % mod

    for b in range(2, b_limit + 1, 2):
        half_b = b // 2
        a_limit = (icbrt_local(limit // (2 * b)) - half_b) // 2
        cf_2b = cf[2 * b]
        # Since b is even, coprime a must be odd.
        for a in range(1, a_limit + 1, 2):
            if gcd(a, b) != 1 or cf_2b == cf[a]:
                continue

            x_base = half_b + 2 * a
            x = 2 * b * x_base * x_base * x_base
            y_base = a - 2 * b
            y_abs = abs(a * y_base * y_base * y_base)
            max_coord = x if x >= y_abs else y_abs
            if max_coord > limit:
                continue

            tmax = isqrt(limit // max_coord)
            total = (total + (x + y_abs) * sumsq_local(tmax)) % mod

    return total


def _self_test() -> None:
    assert H_mod(10**3) == 2535


def main() -> None:
    _self_test()
    print(H_mod(N) % MOD)


if __name__ == "__main__":
    main()
