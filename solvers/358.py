#!/usr/bin/env python
from __future__ import annotations

import math


PREFIX_SCALE = 10**11
PREFIX_VALUE = 137
SUFFIX_VALUE = 56789
SUFFIX_MODULUS = 10**5


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    if n % 3 == 0:
        return n == 3
    limit = math.isqrt(n)
    d = 5
    step = 2
    while d <= limit:
        if n % d == 0:
            return False
        d += step
        step = 6 - step
    return True


def distinct_prime_factors(n: int) -> list[int]:
    factors = []
    if n % 2 == 0:
        factors.append(2)
        while n % 2 == 0:
            n //= 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 2
    if n > 1:
        factors.append(n)
    return factors


def is_full_reptend_prime(p: int) -> bool:
    if not is_prime(p) or p in (2, 5):
        return False
    return all(
        pow(10, (p - 1) // factor, p) != 1
        for factor in distinct_prime_factors(p - 1)
    )


def candidate_prime() -> int:
    low = PREFIX_SCALE // (PREFIX_VALUE + 1) + 1
    high = PREFIX_SCALE // PREFIX_VALUE

    residue = (-pow(SUFFIX_VALUE, -1, SUFFIX_MODULUS)) % SUFFIX_MODULUS
    p = low + (residue - low) % SUFFIX_MODULUS
    while p <= high:
        if PREFIX_SCALE // p == PREFIX_VALUE and is_full_reptend_prime(p):
            return p
        p += SUFFIX_MODULUS
    raise RuntimeError("no matching full reptend prime found")


def solve() -> int:
    p = candidate_prime()
    return 9 * (p - 1) // 2


def main() -> None:
    assert is_full_reptend_prime(7)
    print(solve())


if __name__ == "__main__":
    main()
