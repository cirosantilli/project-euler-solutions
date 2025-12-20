#!/usr/bin/env python3
"""Project Euler 908: Clock Sequence II

A clock sequence is a periodic sequence of positive integers that can be broken into
contiguous segments such that the sum of the n-th segment is n.

We count different clock sequences with (minimal) period at most N.

This program computes C(10^4) mod 1111211113.

No external libraries are used.
"""

import sys

MOD = 1111211113
N = 10_000


def sieve_primes(limit: int):
    """Return list of primes <= limit."""
    if limit < 2:
        return []
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[0:2] = b"\x00\x00"
    import math

    for p in range(2, int(math.isqrt(limit)) + 1):
        if sieve[p]:
            step = p
            start = p * p
            sieve[start : limit + 1 : step] = b"\x00" * (((limit - start) // step) + 1)
    return [i for i in range(limit + 1) if sieve[i]]


def mobius_upto(n: int):
    """Compute mu[0..n] via linear sieve."""
    mu = [0] * (n + 1)
    primes = []
    is_comp = [False] * (n + 1)
    mu[1] = 1
    for i in range(2, n + 1):
        if not is_comp[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            v = i * p
            if v > n:
                break
            is_comp[v] = True
            if i % p == 0:
                mu[v] = 0
                break
            mu[v] = -mu[i]
    return mu


def k_prime_power(p: int, e: int) -> int:
    """k(p^e) = number of distinct triangular residues modulo p^e."""
    if e <= 0:
        return 1
    if p == 2:
        return 1 << e

    # For odd primes, use the proven recurrence (see README).
    k = (p + 1) // 2  # e = 1
    for exp in range(2, e + 1):
        if exp % 2 == 0:
            k = p * k - (p - 1)
        else:
            k = p * k - (p - 1) // 2
    return k


def generate_moduli(max_k: int):
    """Generate all pairs (m, k(m)) with k(m) <= max_k.

    We build m multiplicatively as a product of prime powers, while k(m) is
    multiplicative as well.

    Returns (pairs, max_m).
    """
    primes = sieve_primes(
        2 * max_k
    )  # enough: for odd p, k(p)=(p+1)/2 > max_k when p>2*max_k-1

    # Precompute options for each prime: list of (p^e, k(p^e)) for e>=1 with k<=max_k.
    options = {}
    for p in primes:
        opts = []
        if p == 2:
            m = 2
            k = 2
            while k <= max_k:
                opts.append((m, k))
                m <<= 1
                k <<= 1
        else:
            m = p
            k = (p + 1) // 2
            e = 1
            while k <= max_k:
                opts.append((m, k))
                e += 1
                m *= p
                if e % 2 == 0:
                    k = p * k - (p - 1)
                else:
                    k = p * k - (p - 1) // 2
        options[p] = opts

    pairs = []
    max_m = 1
    sys.setrecursionlimit(1000000)

    def dfs(start_idx: int, m_cur: int, k_cur: int):
        nonlocal max_m
        pairs.append((m_cur, k_cur))
        if m_cur > max_m:
            max_m = m_cur

        # Try extending with primes from start_idx onward
        for j in range(start_idx, len(primes)):
            p = primes[j]
            opts = options[p]
            if not opts:
                continue
            # Smallest k-factor for this prime
            if k_cur * opts[0][1] > max_k:
                break

            for mp, kp in opts:
                k_new = k_cur * kp
                if k_new > max_k:
                    break
                dfs(j + 1, m_cur * mp, k_new)

    dfs(0, 1, 1)
    return pairs, max_m


def prepare_inverses(n: int, mod: int):
    """inv[i] = modular inverse of i modulo mod, for i=1..n (mod is prime)."""
    inv = [0] * (n + 1)
    inv[1] = 1
    for i in range(2, n + 1):
        inv[i] = (mod - (mod // i) * inv[mod % i] % mod) % mod
    return inv


def compute_B(max_period: int, mod: int):
    """Compute B[p] = number of clock sequences with period p (not necessarily minimal)."""
    moduli, _max_m = generate_moduli(max_period)

    inv = prepare_inverses(max_period, mod)

    B = [0] * (max_period + 1)
    B_local = B
    inv_local = inv
    MOD_local = mod

    # For each modulus m with k(m)=k, contribute C(m-k, p-k) to B[p]
    for m, k in moduli:
        if k > max_period:
            continue
        n = m - k
        if n < 0:
            continue
        rmax = max_period - k
        if rmax < 0:
            continue
        if n < rmax:
            rmax = n

        # r = 0
        idx0 = k
        v = B_local[idx0] + 1
        if v >= MOD_local:
            v -= MOD_local
        B_local[idx0] = v

        c = 1
        # r from 1..rmax
        # c = C(n, r) computed iteratively
        for r in range(1, rmax + 1):
            c = (c * (n - r + 1) * inv_local[r]) % MOD_local
            idx = idx0 + r
            v = B_local[idx] + c
            if v >= MOD_local:
                v -= MOD_local
            B_local[idx] = v

    return B


def compute_A_from_B(B, mu, mod: int):
    """A[p] = number of clock sequences with minimal period exactly p."""
    n = len(B) - 1
    A = [0] * (n + 1)
    A_local = A
    B_local = B
    MOD_local = mod

    for d in range(1, n + 1):
        md = mu[d]
        if md == 0:
            continue
        if md == 1:
            for q in range(1, n // d + 1):
                p = d * q
                v = A_local[p] + B_local[q]
                if v >= MOD_local:
                    v -= MOD_local
                A_local[p] = v
        else:  # md == -1
            for q in range(1, n // d + 1):
                p = d * q
                v = A_local[p] - B_local[q]
                if v < 0:
                    v += MOD_local
                A_local[p] = v

    return A


def main():
    B = compute_B(N, MOD)
    mu = mobius_upto(N)
    A = compute_A_from_B(B, mu, MOD)

    # Prefix sums give C(t) for all t <= N
    C = [0] * (N + 1)
    s = 0
    for i in range(1, N + 1):
        s += A[i]
        s %= MOD
        C[i] = s

    # Test values from the problem statement
    assert C[3] == 3
    assert C[4] == 7
    assert C[10] == 561

    print(C[N] % MOD)


if __name__ == "__main__":
    main()
