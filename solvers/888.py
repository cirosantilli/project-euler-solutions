#!/usr/bin/env python3
"""
Project Euler 888 solver (single-file, standard library only).

Game:
- From one pile you may remove 1, 2, 4, or 9 stones, OR split a pile into two non-empty piles.
- Normal play (last move wins).

We count losing positions with m piles, each pile size in [1..N], order irrelevant.
A position is losing iff XOR of pile Grundy numbers is 0.
"""

from __future__ import annotations

import math

# ---------------------------
# Basic number theory helpers
# ---------------------------


def egcd(a: int, b: int):
    """Extended gcd: returns (g,x,y) where g=gcd(a,b) and ax+by=g."""
    x0, y0, x1, y1 = 1, 0, 0, 1
    while b:
        q = a // b
        a, b = b, a - q * b
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    return a, x0, y0


def inv_mod(a: int, mod: int) -> int:
    """Modular inverse of a modulo mod, assuming gcd(a,mod)=1."""
    a %= mod
    g, x, _ = egcd(a, mod)
    if g != 1:
        raise ValueError("inverse does not exist")
    return x % mod


def factor_prime_powers(n: int):
    """Return list of (p, p^e) for prime powers dividing n."""
    out = []
    x = n
    p = 2
    while p * p <= x:
        if x % p == 0:
            pe = 1
            while x % p == 0:
                x //= p
                pe *= p
            out.append((p, pe))
        p = 3 if p == 2 else p + 2
    if x > 1:
        out.append((x, x))
    return out


def crt_setup(mod: int, prime_powers):
    """Precompute CRT constants for pairwise-coprime moduli prime_powers[i][1]."""
    mods = [pe for _, pe in prime_powers]
    Ms = [mod // mi for mi in mods]
    inv_Ms = [inv_mod(Ms[i] % mods[i], mods[i]) for i in range(len(mods))]
    return mods, Ms, inv_Ms


def crt_combine(residues, mods, Ms, inv_Ms, mod_total: int) -> int:
    """Combine residues mod mods[i] into a residue mod mod_total."""
    x = 0
    for r, mi, Mi, invMi in zip(residues, mods, Ms, inv_Ms):
        x = (x + (r % mi) * Mi * invMi) % mod_total
    return x


# ---------------------------
# Grundy computation
# ---------------------------

_REMOVE = (1, 2, 4, 9)


def compute_grundy(limit: int, split_limit: int | None) -> list[int]:
    """
    Compute Grundy numbers g[0..limit].
    If split_limit is not None, only consider splits with one part size <= split_limit.
    """
    g = [0] * (limit + 1)
    for n in range(1, limit + 1):
        seen = 0
        # subtraction moves
        for r in _REMOVE:
            if n >= r:
                seen |= 1 << g[n - r]
        # split moves
        max_i = n - 1 if split_limit is None else min(split_limit, n - 1)
        for i in range(1, max_i + 1):
            seen |= 1 << (g[i] ^ g[n - i])
        # mex
        mex = 0
        while (seen >> mex) & 1:
            mex += 1
        g[n] = mex
    return g


# ---------------------------
# Counting S(N,m) using a ±1 filter on xor (Z2^4)
# ---------------------------


def strip_p(x: int, p: int):
    """Return (x_without_p, v) where v is exponent of p in x."""
    if x == 0:
        return 0, 0
    v = 0
    while x % p == 0:
        x //= p
        v += 1
    return x, v


def precompute_den_tables(m: int, p: int, mod_pe: int):
    """
    For denominator i=1..m, precompute:
      den_free[i] = i with p factors removed
      den_v[i]    = v_p(i)
      inv_den_free[i] mod p^e
    """
    den_free = [0] * (m + 1)
    den_v = [0] * (m + 1)
    inv_free = [0] * (m + 1)
    den_free[0] = 1
    inv_free[0] = 1
    for i in range(1, m + 1):
        f, v = strip_p(i, p)
        den_free[i] = f
        den_v[i] = v
        inv_free[i] = inv_mod(f % mod_pe, mod_pe)  # f is coprime to p
    return den_free, den_v, inv_free


def coeff_term_mod_prime_power(
    N: int, m: int, a: int, p: int, mod_pe: int, den_free, den_v, inv_free
) -> int:
    """
    Compute coefficient of x^m in:
      (1-x)^(-a) * (1+x)^(-(N-a))
    modulo p^e.

    Builds the two negative-binomial series up to degree m, tracking p-adic valuation.
    """
    A = a
    B = N - a

    # u_i = C(A+i-1,i), with convention A=0 -> u_0=1, u_i=0 for i>0
    u = [0] * (m + 1)
    u[0] = 1 % mod_pe
    if A != 0:
        res = 1 % mod_pe
        exp = 0
        for i in range(1, m + 1):
            num = A + i - 1
            num_free, v_num = strip_p(num, p)
            exp += v_num
            exp -= den_v[i]
            if exp < 0:
                raise AssertionError("negative p-valuation encountered")
            res = (res * (num_free % mod_pe)) % mod_pe
            res = (res * inv_free[i]) % mod_pe
            u[i] = (res * pow(p, exp, mod_pe)) % mod_pe

    # w_i = (-1)^i * C(B+i-1,i), with convention B=0 -> w_0=1, w_i=0 for i>0
    w = [0] * (m + 1)
    w[0] = 1 % mod_pe
    if B != 0:
        res = 1 % mod_pe
        exp = 0
        for i in range(1, m + 1):
            num = B + i - 1
            num_free, v_num = strip_p(num, p)
            exp += v_num
            exp -= den_v[i]
            if exp < 0:
                raise AssertionError("negative p-valuation encountered")
            res = (res * (num_free % mod_pe)) % mod_pe
            res = (res * inv_free[i]) % mod_pe
            val = (res * pow(p, exp, mod_pe)) % mod_pe
            if i & 1:
                val = (-val) % mod_pe
            w[i] = val

    # convolution
    out = 0
    for i in range(0, m + 1):
        out = (out + u[i] * w[m - i]) % mod_pe
    return out


def grundy_counts_up_to(
    N: int, g: list[int], pre: int, period: int, max_g: int = 16
) -> list[int]:
    """Count how many n in [1..N] have each Grundy value (0..max_g-1), using periodicity."""
    counts = [0] * max_g
    if N <= 0:
        return counts

    if N < pre:
        for n in range(1, N + 1):
            counts[g[n]] += 1
        return counts

    # prefix
    for n in range(1, pre):
        counts[g[n]] += 1

    # periodic part
    per_counts = [0] * max_g
    for n in range(pre, pre + period):
        per_counts[g[n]] += 1

    total_period_terms = N - pre + 1
    q, r = divmod(total_period_terms, period)

    for k in range(max_g):
        counts[k] += per_counts[k] * q

    for n in range(pre, pre + r):
        counts[g[n]] += 1

    return counts


def S_mod(N: int, m: int, mod: int, g: list[int], pre: int, period: int) -> int:
    """
    Compute S(N,m) modulo mod.

    Uses a 16-point ±1 filter over 4 bits (since Grundy values stay small).
    """
    max_g = 16
    counts = grundy_counts_up_to(N, g, pre, period, max_g=max_g)

    # Factor modulus and prep CRT.
    prime_powers = factor_prime_powers(mod)
    mods, Ms, inv_Ms = crt_setup(mod, prime_powers)

    # Precompute denominator tables per prime power.
    den_tables = []
    for p, pe in prime_powers:
        den_tables.append((p, pe, *precompute_den_tables(m, p, pe)))

    inv16 = inv_mod(16, mod)

    total = 0
    # Each mask s in [0..15] represents which bits are tested.
    # For a Grundy value gv, its character value is (-1)^(popcount(gv & s)).
    for s in range(16):
        a = 0
        for gv, c in enumerate(counts):
            if c and ((gv & s).bit_count() & 1) == 0:
                a += c

        residues = []
        for p, pe, den_free, den_v, inv_free in den_tables:
            residues.append(
                coeff_term_mod_prime_power(N, m, a, p, pe, den_free, den_v, inv_free)
            )

        coef = crt_combine(residues, mods, Ms, inv_Ms, mod)
        total = (total + coef) % mod

    return (total * inv16) % mod


# ---------------------------
# Exact evaluation for statement checks
# ---------------------------


def coeff_term_exact(N: int, m: int, a: int) -> int:
    """
    Exact coefficient of x^m in (1-x)^(-a) * (1+x)^(-(N-a)).

    Series:
      (1-x)^(-t) = sum_{k>=0} C(t+k-1, k) x^k, with t=0 meaning 1
      (1+x)^(-t) = sum_{k>=0} (-1)^k C(t+k-1, k) x^k, with t=0 meaning 1
    """
    A = a
    B = N - a

    def c_multiset(t: int, k: int) -> int:
        if k == 0:
            return 1
        if t == 0:
            return 0
        return math.comb(t + k - 1, k)

    out = 0
    for i in range(0, m + 1):
        out += c_multiset(A, i) * ((-1) ** (m - i)) * c_multiset(B, m - i)
    return out


def S_exact(N: int, m: int) -> int:
    """Exact S(N,m) for small N,m (used for problem-provided checks)."""
    g = compute_grundy(N, split_limit=None)
    counts = [0] * 16
    for n in range(1, N + 1):
        counts[g[n]] += 1

    total = 0
    for s in range(16):
        a = 0
        for gv, c in enumerate(counts):
            if c and ((gv & s).bit_count() & 1) == 0:
                a += c
        total += coeff_term_exact(N, m, a)

    assert total % 16 == 0
    return total // 16


def main():
    # Problem constants
    N = 12491249
    m = 1249
    MOD = 912491249

    # Problem statement checks
    assert S_exact(12, 4) == 204
    assert S_exact(124, 9) == 2259208528408

    # Grundy periodicity parameters (validated below after computation)
    PRE = 322
    PERIOD = 11060
    SPLIT_LIMIT = 600

    # Precompute enough Grundy values to validate one full period repeat.
    precomp_limit = PRE + 2 * PERIOD

    g_fast = compute_grundy(precomp_limit, split_limit=SPLIT_LIMIT)

    # Validate split_limit on a manageable prefix against full splits.
    check_limit = 2000
    g_full = compute_grundy(check_limit, split_limit=None)
    assert g_fast[: check_limit + 1] == g_full

    # Grundy values should fit in 4 bits for the 16-point filter.
    assert max(g_fast) < 16

    # Validate observed periodicity over the computed window.
    for n in range(PRE, precomp_limit - PERIOD + 1):
        assert g_fast[n] == g_fast[n + PERIOD]

    ans = S_mod(N, m, MOD, g_fast, PRE, PERIOD)
    print(ans)


if __name__ == "__main__":
    main()
