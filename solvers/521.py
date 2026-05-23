#!/usr/bin/env python
"""
Project Euler 521: Smallest Prime Factor

The solver keeps the count and sum of numbers that have not yet been removed
by smaller prime factors.  When a prime p is processed, all composites whose
smallest prime factor is p are counted once and then removed from the filtered
tables.
"""

from array import array
from math import isqrt


MOD = 1_000_000_000
TARGET_N = 10**12


def smpf_single(x: int) -> int:
    """Smallest prime factor, used only for tiny checks."""
    if x % 2 == 0:
        return 2
    p = 3
    while p * p <= x:
        if x % p == 0:
            return p
        p += 2
    return x


def initial_sum(x: int) -> int:
    """Return 2 + 3 + ... + x modulo MOD."""
    if x < 2:
        return 0
    return (x * (x + 1) // 2 - 1) % MOD


def filtered_smpf_sum(n: int) -> int:
    if n < 2:
        return 0

    root = isqrt(n)

    count_small = array("Q", [0]) * (root + 1)
    sum_small = array("I", [0]) * (root + 1)
    for x in range(1, root + 1):
        count_small[x] = x - 1
        sum_small[x] = initial_sum(x)

    count_large = array("Q", [0]) * (root + 1)
    sum_large = array("I", [0]) * (root + 1)
    for d in range(1, root + 1):
        x = n // d
        count_large[d] = x - 1
        sum_large[d] = initial_sum(x)

    answer = 0
    n_local = n
    root_local = root
    mod = MOD
    count_s = count_small
    sum_s = sum_small
    count_l = count_large
    sum_l = sum_large

    for p in range(2, root_local + 1):
        count_before = count_s[p - 1]
        if count_s[p] == count_before:
            continue

        sum_before = sum_s[p - 1]
        answer = (answer + p * ((count_l[p] - count_before) % mod)) % mod

        p2 = p * p
        large_stop = n_local // p2
        if large_stop > root_local:
            large_stop = root_local

        for d in range(1, large_stop + 1):
            q = n_local // (d * p)
            if q <= root_local:
                q_count = count_s[q]
                q_sum = sum_s[q]
            else:
                idx = n_local // q
                q_count = count_l[idx]
                q_sum = sum_l[idx]

            count_l[d] -= q_count - count_before
            sum_l[d] = (sum_l[d] - p * ((q_sum - sum_before) % mod)) % mod

        if p2 <= root_local:
            for x in range(root_local, p2 - 1, -1):
                q = x // p
                count_s[x] -= count_s[q] - count_before
                sum_s[x] = (sum_s[x] - p * ((sum_s[q] - sum_before) % mod)) % mod

    return (answer + sum_l[1]) % mod


def main() -> None:
    assert smpf_single(91) == 7
    assert smpf_single(45) == 3
    assert filtered_smpf_sum(100) == 1257
    print(filtered_smpf_sum(TARGET_N))


if __name__ == "__main__":
    main()
