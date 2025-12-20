#!/usr/bin/env python3
"""Project Euler 843: Periodic Circles

This is an optimized pure-Python solution (no external libraries).

Core reductions
--------------
For 0/1 values, the absolute difference equals XOR, so the update is
    b[i] = a[i-1] XOR a[i+1]
which is Rule 90 cellular automaton on a ring. This is linear over GF(2).

Represent the state as a polynomial A(x) in GF(2)[x]/(x^n-1). One update is
multiplication by g(x) = x + x^{-1}.

Possible eventual periods come from the invertible part of the linear map:
for each divisor q(x) of x^n-1 with gcd(g,q)=1, the period is an order of g
modulo q (and for distinct factors, periods combine via LCM).

Performance improvements vs a naive field-order approach
-------------------------------------------------------
Let n = 2^a * m with m odd. In characteristic 2:
    x^n - 1 = x^n + 1 = (x^m + 1)^(2^a)
So we factor x^m+1 into irreducibles p(x), and consider prime powers p^t.

Key speedup: for each irreducible p, we need the order of
    y = x + x^{-1}
mod p. Although p may have degree up to ~n, the element y lies in a smaller
subfield. Its subfield degree k is the Frobenius orbit length:
    smallest k>0 such that y^(2^k) = y.
Then ord(y) divides 2^k-1, and for n<=100 we always have k<=41, so factoring
2^k-1 is cheap.

We also avoid an exponential blow-up in combining prime powers by splitting
periods into an odd part and a 2-power part.

Asserts required by the problem statement:
    S(6)  = 6
    S(30) = 20381

Prints:
    S(100)
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple


# ----------------------------
# GF(2) polynomials as bitmasks
# bit i == coefficient of x^i
# ----------------------------


def poly_deg(a: int) -> int:
    return a.bit_length() - 1


def poly_mod(a: int, mod: int) -> int:
    """a mod mod, with mod assumed non-zero and monic."""
    md = poly_deg(mod)
    while a and poly_deg(a) >= md:
        a ^= mod << (poly_deg(a) - md)
    return a


def poly_mul(a: int, b: int) -> int:
    """Carry-less multiply over GF(2)."""
    res = 0
    while b:
        lsb = b & -b
        shift = lsb.bit_length() - 1
        res ^= a << shift
        b ^= lsb
    return res


def poly_mul_mod(a: int, b: int, mod: int) -> int:
    return poly_mod(poly_mul(a, b), mod)


# fast squaring via byte lookup table
_SQ_TABLE = [0] * 256
for _b in range(256):
    v = 0
    for i in range(8):
        if (_b >> i) & 1:
            v |= 1 << (2 * i)
    _SQ_TABLE[_b] = v


def poly_square(a: int) -> int:
    """Square in GF(2)[x] (inserts zeros between bits)."""
    res = 0
    i = 0
    while a:
        res |= _SQ_TABLE[a & 0xFF] << (16 * i)
        a >>= 8
        i += 1
    return res


def poly_square_mod(a: int, mod: int) -> int:
    return poly_mod(poly_square(a), mod)


def poly_pow_mod(a: int, exp: int, mod: int) -> int:
    res = 1
    base = poly_mod(a, mod)
    while exp > 0:
        if exp & 1:
            res = poly_mul_mod(res, base, mod)
        exp >>= 1
        if exp:
            base = poly_square_mod(base, mod)
    return res


def poly_gcd(a: int, b: int) -> int:
    while b:
        a, b = b, poly_mod(a, b)
    return a


def poly_div_exact(a: int, b: int) -> int:
    """Return a/b assuming exact division in GF(2)[x]."""
    if b == 0:
        raise ZeroDivisionError
    db = poly_deg(b)
    q = 0
    while a and poly_deg(a) >= db:
        shift = poly_deg(a) - db
        q ^= 1 << shift
        a ^= b << shift
    assert a == 0
    return q


def ilcm(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return a // math.gcd(a, b) * b


# ----------------------------
# Small integer factoring (we only need up to 2^41-1)
# ----------------------------

MAX_K = 41
MAX_MERSENNE = (1 << MAX_K) - 1
SIEVE_LIMIT = int(math.isqrt(MAX_MERSENNE)) + 1


def sieve_primes(limit: int) -> List[int]:
    bs = bytearray(b"\x01") * (limit + 1)
    bs[:2] = b"\x00\x00"
    for p in range(2, int(limit**0.5) + 1):
        if bs[p]:
            step = p
            start = p * p
            bs[start : limit + 1 : step] = b"\x00" * (((limit - start) // step) + 1)
    return [i for i in range(limit + 1) if bs[i]]


_PRIMES = sieve_primes(SIEVE_LIMIT)


def factor_small(n: int) -> List[int]:
    """Prime factors of n (with multiplicity), using trial division."""
    res: List[int] = []
    tmp = n
    for p in _PRIMES:
        if p * p > tmp:
            break
        while tmp % p == 0:
            res.append(p)
            tmp //= p
    if tmp > 1:
        res.append(tmp)
    return res


_mersenne_factor_cache: Dict[int, List[int]] = {}


def mersenne_factors(k: int) -> List[int]:
    """Prime factors (with multiplicity) of 2^k - 1, cached."""
    if k in _mersenne_factor_cache:
        return _mersenne_factor_cache[k][:]
    m = (1 << k) - 1
    fac = factor_small(m)
    fac.sort()
    _mersenne_factor_cache[k] = fac
    return fac[:]


# ----------------------------
# Berlekamp factorization for GF(2) polynomials (square-free, monic)
# ----------------------------


def berlekamp_nullspace_basis(f: int) -> List[int]:
    """Nullspace basis of (Q-I) where Q is Frobenius a -> a^2 mod f."""
    n = poly_deg(f)
    rows = [0] * n

    # Column j is x^(2j) mod f, expressed in basis 1..x^(n-1)
    for j in range(n):
        col = poly_mod(1 << (2 * j), f)
        i = 0
        while col:
            if col & 1:
                rows[i] |= 1 << j
            col >>= 1
            i += 1

    # Q - I
    for i in range(n):
        rows[i] ^= 1 << i

    # Gauss-Jordan over GF(2)
    pivot_cols: List[int] = []
    pivot_row_for_col: Dict[int, int] = {}
    r = 0
    for c in range(n):
        pivot = None
        for i in range(r, n):
            if (rows[i] >> c) & 1:
                pivot = i
                break
        if pivot is None:
            continue
        rows[r], rows[pivot] = rows[pivot], rows[r]
        pv = rows[r]
        for i in range(n):
            if i != r and ((rows[i] >> c) & 1):
                rows[i] ^= pv
        pivot_cols.append(c)
        pivot_row_for_col[c] = r
        r += 1
        if r == n:
            break

    pivot_set = set(pivot_cols)
    free_cols = [c for c in range(n) if c not in pivot_set]

    basis: List[int] = []
    for fc in free_cols:
        v = 1 << fc
        for pc in pivot_cols:
            row = rows[pivot_row_for_col[pc]]
            if (row >> fc) & 1:
                v |= 1 << pc
        basis.append(v)
    return basis


def factor_squarefree(f: int) -> List[int]:
    """Factor a square-free monic polynomial over GF(2) into irreducibles."""
    if f == 1:
        return []
    if poly_deg(f) <= 1:
        return [f]

    basis = berlekamp_nullspace_basis(f)
    if len(basis) == 1:
        return [f]

    # Berlekamp splitting with deterministic randomness
    for _ in range(300):
        comb = 0
        for v in basis[1:]:
            if random.getrandbits(1):
                comb ^= v
        if comb in (0, 1):
            continue
        g = poly_gcd(f, comb)
        if g not in (1, f):
            h = poly_div_exact(f, g)
            return factor_squarefree(g) + factor_squarefree(h)
        g = poly_gcd(f, comb ^ 1)
        if g not in (1, f):
            h = poly_div_exact(f, g)
            return factor_squarefree(g) + factor_squarefree(h)

    raise RuntimeError("Berlekamp split failed (unexpected for these sizes).")


_poly_factor_cache: Dict[int, List[int]] = {}


def factor_poly_squarefree_cached(f: int) -> List[int]:
    if f in _poly_factor_cache:
        return _poly_factor_cache[f][:]
    fac = factor_squarefree(f)
    _poly_factor_cache[f] = fac
    return fac[:]


_factor_cache_by_m: Dict[int, List[int]] = {}


def irreducible_factors_xm_plus_1(m: int) -> List[int]:
    """Factor x^m + 1 over GF(2) for odd m. Result is square-free."""
    if m in _factor_cache_by_m:
        return _factor_cache_by_m[m][:]
    f = (1 << m) | 1  # x^m + 1
    fac = factor_poly_squarefree_cached(f)
    _factor_cache_by_m[m] = fac
    return fac[:]


# ----------------------------
# Order computations
# ----------------------------


def frobenius_orbit_degree(a: int, mod_irred: int) -> int:
    """Smallest k>0 such that a^(2^k) == a in GF(2)[x]/(mod_irred)."""
    t = a
    for k in range(1, poly_deg(mod_irred) + 1):
        t = poly_square_mod(t, mod_irred)
        if t == a:
            return k
    # Should never happen
    return poly_deg(mod_irred)


def multiplicative_order(a: int, mod_irred: int) -> int:
    """Order of nonzero a modulo irreducible mod_irred."""
    if a == 0:
        return 0

    k = frobenius_orbit_degree(a, mod_irred)
    group_order = (1 << k) - 1
    if group_order == 1:
        return 1

    order = group_order
    fac = mersenne_factors(k)
    for p in sorted(set(fac)):
        while order % p == 0:
            cand = order // p
            if poly_pow_mod(a, cand, mod_irred) == 1:
                order = cand
            else:
                break
    return order


def max_two_lift_exponent(
    base_poly: int, base_odd_order: int, p: int, max_exp: int
) -> int:
    """For modulus p^t, t=1..max_exp, find max v2(order_t / base_odd_order).

    Important: for t>1 we must use the *original* update polynomial
    g(x)=x+x^(n-1) modulo p^t (we cannot simplify x^(n-1) to x^{-1} there).
    """
    if max_exp <= 1:
        return 0

    order = base_odd_order
    mod = p
    for _t in range(2, max_exp + 1):
        mod = poly_mul(mod, p)  # p^t
        while poly_pow_mod(base_poly, order, mod) != 1:
            order *= 2

    ratio = order // base_odd_order
    # ratio is a power of two
    return ratio.bit_length() - 1


# ----------------------------
# Period enumeration per n
# ----------------------------


def periods_for_n(n: int) -> List[int]:
    """All possible eventual periods for ring size n (sorted)."""
    # n = 2^a * m with m odd
    m = n
    a = 0
    while m % 2 == 0:
        m //= 2
        a += 1
    max_exp = 1 << a

    # The update polynomial in GF(2)[x] is g(x) = x + x^(n-1).
    # Modulo an irreducible factor p of x^m+1 (field case, exponent 1), x^m = 1 holds,
    # hence x^(n-1)=x^(-1). But for higher powers p^t (t>1) that simplification is NOT
    # valid, so we must keep the original g(x) for lifting.
    g_poly = (1 << 1) | (1 << (n - 1))

    # DP over odd parts: map odd_lcm -> max two-exponent achievable
    dp: Dict[int, int] = {1: 0}

    for p in irreducible_factors_xm_plus_1(m):
        # Reduce g modulo p (field element). If g is not invertible modulo p, it only
        # contributes transient (nilpotent) behaviour and cannot influence eventual periods.
        g_mod_p = poly_mod(g_poly, p)
        if poly_gcd(g_mod_p, p) != 1:
            continue

        odd_order = multiplicative_order(g_mod_p, p)  # odd
        smax = max_two_lift_exponent(g_poly, odd_order, p, max_exp)

        # Update dp with choosing this factor (or not choosing it)
        new_dp = dict(dp)
        for cur_l, cur_s in dp.items():
            nl = ilcm(cur_l, odd_order)
            ns = cur_s if cur_s >= smax else smax
            prev = new_dp.get(nl)
            if prev is None or ns > prev:
                new_dp[nl] = ns
        dp = new_dp

    periods = set()
    for odd_l, smax in dp.items():
        # all exponents 0..smax are achievable
        for e in range(smax + 1):
            periods.add(odd_l << e)
    return sorted(periods)


def compute_S_100_with_asserts() -> int:
    all_periods = set()
    s6 = None
    s30 = None

    for n in range(3, 101):
        all_periods.update(periods_for_n(n))
        if n == 6:
            s6 = sum(all_periods)
        if n == 30:
            s30 = sum(all_periods)

    assert s6 == 6
    assert s30 == 20381
    return sum(all_periods)


def main() -> None:
    # Deterministic Berlekamp splitting
    random.seed(0)

    print(compute_S_100_with_asserts())


if __name__ == "__main__":
    main()
