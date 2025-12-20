#!/usr/bin/env python3
"""Project Euler 937: Equiproduct Partition

This program computes G(10^8) modulo 1_000_000_007.

No external libraries are used.
"""

from __future__ import annotations

import math
import sys
from array import array

MOD = 1_000_000_007


def _is_relevant_odd_prime(p: int) -> bool:
    # Relevant odd primes are inert in Z[sqrt(-2)]: p % 8 in {5, 7}.
    return (p & 7) == 5 or (p & 7) == 7


def _update_hash_prime_exp(
    p: int, add: int, keys: array, exps: array, pars: bytearray, mask: int
) -> int:
    """Add to the exponent of prime p in a custom open-addressing table.

    Stores:
      keys[i] = prime (0 means empty)
      exps[i] = exponent in current factorial
      pars[i] = parity(popcount(exps[i])) in {0,1}

    Returns 1 iff parity(popcount(exponent)) changed.
    """
    # Multiplicative hash, mask is (size-1) with size power-of-two.
    i = (p * 2654435761) & mask
    while True:
        k = keys[i]
        if k == 0:
            keys[i] = p
            # old exp/par are 0
            old_par = 0
            old_exp = 0
            break
        if k == p:
            old_exp = exps[i]
            old_par = pars[i]
            break
        i = (i + 1) & mask

    new_exp = old_exp + add
    exps[i] = new_exp
    new_par = new_exp.bit_count() & 1
    if new_par != old_par:
        pars[i] = new_par
        return 1
    return 0


def _pick_table_size(n: int) -> int:
    """Pick power-of-two table size for relevant primes <= n."""
    if n < 50:
        return 1 << 8
    # crude overestimate: about half of primes are relevant; pi(n) ~ n/log n.
    est = int(n / max(2.0 * math.log(n), 1.0)) + 16
    # keep load factor comfortably below ~0.7
    need = est * 2
    size = 1
    while size < need:
        size <<= 1
    return max(size, 1 << 10)


def _build_spf_odd_u16(n: int) -> array:
    """Smallest prime factor for odd numbers up to n.

    Index i represents odd m = 2*i + 1.
    spf[i] = 0 for primes (and for 1), otherwise the smallest odd prime factor.

    For n <= 1e8, smallest odd prime factor of any composite is <= sqrt(n) <= 10000,
    which fits in unsigned 16-bit.
    """
    size = (n // 2) + 1
    spf = array("H", [0]) * size

    limit = int(math.isqrt(n))
    # iterate odd primes p
    for p in range(3, limit + 1, 2):
        if spf[p // 2] == 0:  # p is prime
            step = p << 1  # 2p
            start = p * p
            for m in range(start, n + 1, step):
                idx = m // 2
                if spf[idx] == 0:
                    spf[idx] = p

    return spf


def compute_G_naive(n: int, mod: int = MOD) -> int:
    """Reference implementation for small n (trial division + dict)."""
    fact = 1
    ans = 0

    # Track parity(popcount(exponent)) for relevant primes only.
    exps: dict[int, int] = {}
    global_par = 0  # 0 => in A, 1 => in B

    exp2 = 0
    par2 = 0

    for k in range(1, n + 1):
        fact = (fact * k) % mod

        x = k
        if (x & 1) == 0:
            # v2(x) = trailing zeros
            tz = (x & -x).bit_length() - 1
            if tz:
                exp2_new = exp2 + tz
                par2_new = exp2_new.bit_count() & 1
                if par2_new != par2:
                    global_par ^= 1
                    par2 = par2_new
                exp2 = exp2_new
                x >>= tz

        d = 3
        while d * d <= x:
            if x % d == 0:
                e = 0
                while x % d == 0:
                    x //= d
                    e += 1
                if _is_relevant_odd_prime(d):
                    old = exps.get(d, 0)
                    new = old + e
                    exps[d] = new
                    if (old.bit_count() ^ new.bit_count()) & 1:
                        global_par ^= 1
            d += 2

        if x > 1 and _is_relevant_odd_prime(x):
            old = exps.get(x, 0)
            new = old + 1
            exps[x] = new
            if (old.bit_count() ^ new.bit_count()) & 1:
                global_par ^= 1

        if global_par == 0:
            ans += fact
            ans %= mod

    return ans


def compute_G_fast(n: int, mod: int = MOD) -> int:
    """Fast path intended for n up to 1e8.

    Uses:
      * smallest-prime-factor sieve for odd numbers (u16)
      * custom open-addressing hash table for relevant prime exponents
      * incremental factorial modulo mod

    Note: still a large computation; designed to be memory-aware and reasonably fast
    in pure Python.
    """

    spf = _build_spf_odd_u16(n)

    # Hash table to store exponents/parities for relevant odd primes only.
    size = _pick_table_size(n)
    mask = size - 1
    keys = array("I", [0]) * size
    exps = array("I", [0]) * size
    pars = bytearray(size)

    fact = 1
    ans = 0

    global_par = 0  # 0 => in A, 1 => in B

    exp2 = 0
    par2 = 0

    for k in range(1, n + 1):
        fact = (fact * k) % mod

        x = k
        if (x & 1) == 0:
            tz = (x & -x).bit_length() - 1
            if tz:
                exp2_new = exp2 + tz
                par2_new = exp2_new.bit_count() & 1
                if par2_new != par2:
                    global_par ^= 1
                    par2 = par2_new
                exp2 = exp2_new
                x >>= tz

        while x > 1:
            p = spf[x >> 1]
            if p == 0:
                # x is prime
                if _is_relevant_odd_prime(x):
                    global_par ^= _update_hash_prime_exp(x, 1, keys, exps, pars, mask)
                break

            # count exponent of p in x
            e = 1
            x //= p
            while x % p == 0:
                x //= p
                e += 1

            if _is_relevant_odd_prime(p):
                global_par ^= _update_hash_prime_exp(p, e, keys, exps, pars, mask)

        if global_par == 0:
            ans += fact
            if ans >= mod:
                ans -= mod

    return ans


def main() -> None:
    # Problem statement test values.
    assert compute_G_fast(4) == 25
    assert compute_G_fast(7) == 745
    assert compute_G_fast(100) % MOD == 709772949

    n = 100_000_000
    if len(sys.argv) >= 2:
        n = int(sys.argv[1])
    print(compute_G_fast(n) % MOD)


if __name__ == "__main__":
    main()
