#!/usr/bin/env python3
"""
Project Euler 772 - Balanceable k-bounded Partitions

We use the fact that:
    f(k) = 2 * lcm(1,2,...,k)

The program computes f(10^8) modulo 1_000_000_007 (or for a user-provided k).

No external libraries are used.
"""

from __future__ import annotations

import math
import sys

MOD = 1_000_000_007


def _odd_primes_upto(limit: int) -> list[int]:
    """Return all odd primes <= limit using an odd-only sieve."""
    if limit < 3:
        return []
    # index i represents the odd number (2*i + 1)
    sieve = bytearray(limit // 2 + 1)
    r = int(math.isqrt(limit))
    size = len(sieve)

    for p in range(3, r + 1, 2):
        if sieve[p // 2]:
            continue
        start = p * p
        start_idx = start // 2
        step = p
        sieve[start_idx::step] = b"\x01" * (((size - start_idx - 1) // step) + 1)

    return [p for p in range(3, limit + 1, 2) if not sieve[p // 2]]


def lcm_1_to_k_mod(k: int, mod: int = MOD) -> int:
    """
    Compute lcm(1..k) modulo mod.

    lcm(1..k) = Π_{p prime <= k} p^{floor(log_p(k))}
    """
    if k <= 1:
        return 1 % mod

    res = 1

    # Prime 2 handled separately (we sieve only odds later).
    e = 0
    v = 2
    while v <= k:
        e += 1
        v <<= 1
    res = (res * pow(2, e, mod)) % mod

    sqrt_k = int(math.isqrt(k))
    base_primes = _odd_primes_upto(sqrt_k)

    # Segmented sieve over odd numbers in [3..k]
    segment_odds = 1_000_000  # number of odd candidates per segment (~1MB)
    span = 2 * segment_odds  # integer range covered by one segment

    low = 3
    while low <= k:
        high = low + span
        if high > k + 1:
            high = k + 1

        # odd numbers in [low, high): low, low+2, ..., < high
        count = (high - low + 1) // 2
        seg = bytearray(count)  # 0 = prime candidate, 1 = composite

        for p in base_primes:
            p2 = p * p
            if p2 >= high:
                break
            start = p2 if p2 >= low else ((low + p - 1) // p) * p
            if (start & 1) == 0:
                start += p
            if start >= high:
                continue
            idx = (start - low) // 2
            step = p
            seg[idx::step] = b"\x01" * (((count - idx - 1) // step) + 1)

        n = low
        for is_comp in seg:
            if not is_comp:
                # n is an odd prime
                res = (res * n) % mod
                if n <= sqrt_k:
                    # multiply the extra powers for this prime: p^(e-1)
                    power = n * n
                    while power <= k:
                        res = (res * n) % mod
                        power *= n
            n += 2

        # next segment start: first odd >= high
        low = high if (high & 1) else high + 1

    return res


def f(k: int, mod: int = MOD) -> int:
    """Return f(k) modulo mod."""
    return (2 * lcm_1_to_k_mod(k, mod)) % mod


def _self_test() -> None:
    # Test values from the problem statement
    assert f(3) == 12
    assert f(30) == 179092994


def main() -> None:
    _self_test()
    k = 10**8
    if len(sys.argv) > 1:
        k = int(sys.argv[1])
    print(f(k))


if __name__ == "__main__":
    main()
