#!/usr/bin/env python3
"""
Project Euler 858: LCM

Define G(N) = sum over all subsets S of {1..N} of lcm(S), with lcm(empty)=1.
Compute G(800) modulo 1_000_000_007.

This implementation:
- Uses the identity: n = sum_{d|n} phi(d).
- Rewrites the subset-lcm sum into an inclusion-exclusion over chosen prime powers.
- Splits primes into:
    * small primes p <= sqrt(N) (few, brute force over exponent choices)
    * large primes p  > sqrt(N) (many, but independent due to disjoint multiples)
- Uses bitmasks to track coverage quickly.
- Memoizes the large-prime product based only on a small multiplier-mask state.

No external libraries are used.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

MOD = 1_000_000_007


def primes_up_to(n: int) -> List[int]:
    """Sieve of Eratosthenes."""
    if n < 2:
        return []
    sieve = bytearray(b"\x01") * (n + 1)
    sieve[0:2] = b"\x00\x00"
    limit = int(n**0.5)
    for p in range(2, limit + 1):
        if sieve[p]:
            step = p
            start = p * p
            sieve[start : n + 1 : step] = b"\x00" * (((n - start) // step) + 1)
    return [i for i in range(2, n + 1) if sieve[i]]


def brute_exact_g(n: int) -> int:
    """
    Exact G(n) for small n using DP over distinct LCM values.
    Keeps a map lcm_value -> number of subsets giving it.
    """
    from math import gcd

    counts: Dict[int, int] = {1: 1}  # empty subset
    for a in range(1, n + 1):
        new_counts = dict(counts)  # subsets not taking a
        for l, c in counts.items():
            l2 = l * a // gcd(l, a)
            new_counts[l2] = new_counts.get(l2, 0) + c
        counts = new_counts
    return sum(l * c for l, c in counts.items())


def solve(n: int) -> int:
    primes = primes_up_to(n)
    sqrt_n = math.isqrt(n)

    # Highest exponent e_p such that p^e_p <= n (for L = lcm(1..n)).
    e: Dict[int, int] = {}
    p_pow_e_mod: Dict[int, int] = {}
    inv_p_pow_e: Dict[int, int] = {}

    L_mod = 1
    for p in primes:
        ep = 0
        t = p
        while t <= n:
            ep += 1
            t *= p
        e[p] = ep
        pe_mod = pow(p, ep, MOD)
        p_pow_e_mod[p] = pe_mod
        inv_p_pow_e[p] = pow(pe_mod, MOD - 2, MOD)
        L_mod = (L_mod * pe_mod) % MOD

    small_primes = [p for p in primes if p <= sqrt_n]
    large_primes = [p for p in primes if p > sqrt_n]

    # For large primes p > sqrt(n), any multiple is p*k where k <= n/p <= sqrt(n).
    # Max n/p occurs at smallest possible p > sqrt(n), so:
    Kmax = n // (sqrt_n + 1) if n >= 2 else 0

    # Precompute powers of 2 and inverse powers of 2.
    pow2 = [1] * (n + 1)
    for i in range(1, n + 1):
        pow2[i] = (pow2[i - 1] * 2) % MOD

    inv2 = (MOD + 1) // 2
    inv2pow = [1] * (Kmax + 1)
    for i in range(1, Kmax + 1):
        inv2pow[i] = (inv2pow[i - 1] * inv2) % MOD

    prefix_masks = [(1 << t) - 1 for t in range(Kmax + 1)]

    # Options for each small prime:
    #   r=0 means "not selected"
    #   r>=1 means select q=p^r, with weight = -phi(p^r)/p^{e_p}
    # Also store coverage masks:
    #   maskN: numbers <= n covered by multiples of q
    #   maskK: multipliers <= Kmax covered by multiples of q
    options: Dict[int, List[Tuple[int, int, int]]] = {}
    for p in small_primes:
        opts: List[Tuple[int, int, int]] = [(0, 0, 1)]  # not selected
        ep = e[p]
        for r in range(1, ep + 1):
            q = p**r

            maskN = 0
            for m in range(q, n + 1, q):
                maskN |= 1 << (m - 1)

            maskK = 0
            if q <= Kmax:
                for m in range(q, Kmax + 1, q):
                    maskK |= 1 << (m - 1)

            # phi(p^r) = p^(r-1) * (p-1)
            phi = (pow(p, r - 1, MOD) * (p - 1)) % MOD
            w = (phi * inv_p_pow_e[p]) % MOD
            weight = MOD - w  # includes the (-1) sign
            opts.append((maskN, maskK, weight))
        options[p] = opts

    # For large primes: only exponent 1 is possible and e_p=1, so weight is -phi(p)/p = -(p-1)/p.
    w_large: Dict[int, int] = {}
    inv_p: Dict[int, int] = {}
    for p in large_primes:
        invp = pow(p, MOD - 2, MOD)
        inv_p[p] = invp
        w_large[p] = ((p - 1) * invp) % MOD  # (p-1)/p

    # Memoize large-prime product depending only on maskK.
    large_cache: Dict[int, int] = {}

    def large_product(maskK: int) -> int:
        if maskK in large_cache:
            return large_cache[maskK]
        prod = 1
        for p in large_primes:
            t = n // p  # <= Kmax
            covered = (maskK & prefix_masks[t]).bit_count() if t > 0 else 0
            new = t - covered
            # Factor = 1 + (-w_large[p]) * 2^{-new} = 1 - w_large[p]*inv2pow[new]
            prod = (prod * (1 - (w_large[p] * inv2pow[new]) % MOD)) % MOD
        large_cache[maskK] = prod
        return prod

    total = 0
    sp_list = small_primes

    def dfs(i: int, maskN: int, maskK: int, coeff: int) -> None:
        nonlocal total
        if i == len(sp_list):
            covered_count = maskN.bit_count()
            base = pow2[n - covered_count]
            lp = large_product(maskK)
            total = (total + coeff * base % MOD * lp) % MOD
            return
        p = sp_list[i]
        for addN, addK, w in options[p]:
            dfs(i + 1, maskN | addN, maskK | addK, (coeff * w) % MOD)

    dfs(0, 0, 0, 1)

    return (L_mod * total) % MOD


def main() -> None:
    # Asserts from problem statement (exact values).
    assert brute_exact_g(5) == 528
    assert brute_exact_g(20) == 8463108648960

    print(solve(800))


if __name__ == "__main__":
    main()
