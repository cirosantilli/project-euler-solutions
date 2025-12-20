#!/usr/bin/env python3
"""Project Euler 913: Row-major vs Column-major

Let S(n, m) be the minimal number of swaps (swapping any two entries at a time)
needed to transform an n x m matrix containing 1..nm from row-major order to
column-major order.

This program:
  * asserts the provided examples:
      - S(3, 4) = 8
      - sum_{2 <= n <= m <= 100} S(n, m) = 12578833
  * computes and prints:
      - sum_{2 <= n <= m <= 100} S(n^4, m^4)

No external libraries are used (only Python standard library).
"""

from __future__ import annotations

from math import gcd
from typing import Dict, List, Tuple


# -------------------------
# Small prime utilities
# -------------------------


def sieve_primes(limit: int) -> List[int]:
    """Return list of primes <= limit via sieve."""
    if limit < 2:
        return []
    is_prime = bytearray(b"\x01") * (limit + 1)
    is_prime[0:2] = b"\x00\x00"
    p = 2
    while p * p <= limit:
        if is_prime[p]:
            step = p
            start = p * p
            is_prime[start : limit + 1 : step] = b"\x00" * (
                ((limit - start) // step) + 1
            )
        p += 1
    return [i for i in range(2, limit + 1) if is_prime[i]]


_PRIMES: List[int] = sieve_primes(100000)  # more than enough for our trial divisions


# -------------------------
# Factorisation with caching
# -------------------------

_FAC_CACHE: Dict[int, Tuple[Tuple[int, int], ...]] = {}


def _factorize_small(n: int) -> Dict[int, int]:
    """Prime factorization by trial division; suited for n up to ~1e8 here."""
    if n <= 1:
        return {}

    cached = _FAC_CACHE.get(n)
    if cached is not None:
        return dict(cached)

    x = n
    out: Dict[int, int] = {}
    for p in _PRIMES:
        if p * p > x:
            break
        if x % p == 0:
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            out[p] = e
    if x > 1:
        out[x] = out.get(x, 0) + 1

    _FAC_CACHE[n] = tuple(sorted(out.items()))
    return out


def _merge_factorizations(facs: List[Dict[int, int]]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for f in facs:
        for p, e in f.items():
            out[p] = out.get(p, 0) + e
    return out


_T4M1_CACHE: Dict[int, Tuple[Tuple[int, int], ...]] = {}


def factorize_t4_minus_1(t: int) -> Dict[int, int]:
    """Factorize t^4 - 1 = (t-1)(t+1)(t^2+1) using trial division on each factor.

    With t <= 10000 (in our main sum), each factor is <= 100000001, so trial
    division with primes up to 1e5 (and early stop at sqrt) is fast.
    """
    cached = _T4M1_CACHE.get(t)
    if cached is not None:
        return dict(cached)

    a = t - 1
    b = t + 1
    c = t * t + 1
    fac = _merge_factorizations(
        [
            _factorize_small(a),
            _factorize_small(b),
            _factorize_small(c),
        ]
    )

    # Cache
    _T4M1_CACHE[t] = tuple(sorted(fac.items()))
    return fac


# -------------------------
# Multiplicative order
# -------------------------

_LAMBDA_CACHE: Dict[Tuple[int, int], Tuple[int, Tuple[int, ...]]] = {}


def _carmichael_lambda_prime_power(p: int, a: int) -> int:
    if a <= 0:
        return 1
    if p == 2:
        if a == 1:
            return 1
        if a == 2:
            return 2
        return 1 << (a - 2)
    return (p - 1) * (p ** (a - 1))


def _lambda_prime_factors(p: int, a: int) -> Tuple[int, Tuple[int, ...]]:
    """Return (lambda(p^a), unique prime divisors of lambda(p^a))."""
    key = (p, a)
    cached = _LAMBDA_CACHE.get(key)
    if cached is not None:
        return cached

    lam = _carmichael_lambda_prime_power(p, a)
    if lam == 1:
        primes = ()
    elif p == 2:
        primes = (2,)
    else:
        fac = _factorize_small(p - 1)
        if a >= 2:
            fac[p] = fac.get(p, 0) + (a - 1)
        primes = tuple(sorted(fac.keys()))

    _LAMBDA_CACHE[key] = (lam, primes)
    return lam, primes


def multiplicative_order(a: int, mod: int, p: int, exp: int) -> int:
    """Compute ord_mod(a) where mod = p^exp and gcd(a, mod)=1."""
    if mod == 1:
        return 1
    a %= mod
    lam, primes = _lambda_prime_factors(p, exp)
    order = lam
    for q in primes:
        while order % q == 0 and pow(a, order // q, mod) == 1:
            order //= q
    return order


# -------------------------
# Core: compute S(rows, cols)
# -------------------------


def lcm(a: int, b: int) -> int:
    return a // gcd(a, b) * b


def _prime_power_tables(
    mult: int, factors: Dict[int, int]
) -> Tuple[List[int], List[List[int]], List[List[int]]]:
    """For each prime p^e dividing M, build:
      - phi_pows[a] = phi(p^a) for a=0..e
      - ord_pows[a] = ord_{p^a}(mult) for a=0..e
    Return (primes, phi_tables, ord_tables) aligned by index.
    """
    primes = sorted(factors.keys())
    phi_tables: List[List[int]] = []
    ord_tables: List[List[int]] = []

    for p in primes:
        e = factors[p]
        phi_pows = [1] * (e + 1)
        ord_pows = [1] * (e + 1)

        mod = 1
        for a in range(1, e + 1):
            mod *= p
            # phi(p^a)
            if p == 1:
                phi_pows[a] = 1
            else:
                phi_pows[a] = (p - 1) * (p ** (a - 1))
            ord_pows[a] = multiplicative_order(mult, mod, p, a)

        phi_tables.append(phi_pows)
        ord_tables.append(ord_pows)

    return primes, phi_tables, ord_tables


def _sum_phi_over_order(mult: int, factors: Dict[int, int]) -> int:
    """Compute sum_{Q | M} phi(Q) / ord_Q(mult), where M has factorization 'factors'."""
    primes, phi_tables, ord_tables = _prime_power_tables(mult, factors)
    exps = [factors[p] for p in primes]

    # DFS over divisor exponents.
    total = 0

    def dfs(i: int, cur_phi: int, cur_ord: int) -> None:
        nonlocal total
        if i == len(primes):
            total += cur_phi // cur_ord
            return
        phi_pows = phi_tables[i]
        ord_pows = ord_tables[i]
        for a in range(exps[i] + 1):
            dfs(i + 1, cur_phi * phi_pows[a], lcm(cur_ord, ord_pows[a]))

    dfs(0, 1, 1)
    return total


def S_general(n: int, m: int) -> int:
    """Compute S(n, m) for moderate n, m (here n,m <= 100 for tests)."""
    N = n * m
    M = N - 1
    # Factor M directly (small)
    factors = _factorize_small(M)
    mult = m
    sum_contrib = _sum_phi_over_order(mult, factors)
    return N - 1 - sum_contrib


def S_pow4(n: int, m: int) -> int:
    """Compute S(n^4, m^4) for 2 <= n <= m <= 100."""
    t = n * m
    N = t**4
    factors = factorize_t4_minus_1(t)  # factorization of N-1
    mult = m**4
    sum_contrib = _sum_phi_over_order(mult, factors)
    return N - 1 - sum_contrib


# -------------------------
# Verification helpers
# -------------------------


def _brutal_cycles(n: int, m: int) -> int:
    """Brute-force cycle count for small n,m (debug only)."""
    N = n * m
    # mapping f(i) = (i mod n)*m + (i//n)
    f = [(i % n) * m + (i // n) for i in range(N)]
    seen = [False] * N
    cycles = 0
    for i in range(N):
        if not seen[i]:
            cycles += 1
            j = i
            while not seen[j]:
                seen[j] = True
                j = f[j]
    return cycles


def _S_brutal(n: int, m: int) -> int:
    return n * m - _brutal_cycles(n, m)


# -------------------------
# Main
# -------------------------


def main() -> None:
    # Provided examples
    assert S_general(3, 4) == 8

    # Cross-check our formula on a few tiny cases via brute-force.
    for n in range(2, 6):
        for m in range(n, 6):
            assert S_general(n, m) == _S_brutal(n, m)

    # Provided checksum
    s = 0
    for n in range(2, 101):
        for m in range(n, 101):
            s += S_general(n, m)
    assert s == 12578833

    # Actual problem
    ans = 0
    for n in range(2, 101):
        for m in range(n, 101):
            ans += S_pow4(n, m)
    print(ans)


if __name__ == "__main__":
    main()
