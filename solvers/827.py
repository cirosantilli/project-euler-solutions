#!/usr/bin/env python3
"""
Project Euler 827 — Pythagorean Triple Occurrence

We need Q(n): the smallest positive integer that occurs in exactly n Pythagorean
triples (a,b,c) with a<b<c and a^2+b^2=c^2.

This program computes:
    sum_{k=1..18} Q(10^k) mod 409120391

No external libraries are used.
"""

from __future__ import annotations

import math
import random
import decimal
import functools
import sys
from typing import Dict, List, Tuple

MOD = 409120391

# ---------- high-precision log2 for safe comparisons of gigantic integers ----------
decimal.getcontext().prec = 90
_LN2 = decimal.Decimal(2).ln()


def _log2_int(n: int) -> decimal.Decimal:
    return decimal.Decimal(n).ln() / _LN2


# ---------- Miller-Rabin (deterministic for 64-bit) + Pollard Rho factorization ----------
def _is_probable_prime(n: int) -> bool:
    if n < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small_primes:
        if n % p == 0:
            return n == p

    d = n - 1
    s = 0
    while d % 2 == 0:
        s += 1
        d //= 2

    # Deterministic bases for n < 2^64
    for a in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _pollard_rho(n: int) -> int:
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if n % 5 == 0:
        return 5

    while True:
        c = random.randrange(1, n - 1)
        x = random.randrange(0, n - 1)
        y = x
        d = 1

        # f(x) = x^2 + c  (mod n)
        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = math.gcd(abs(x - y), n)

        if d != n:
            return d


_factor_cache: Dict[int, Dict[int, int]] = {}


def _factor(n: int) -> Dict[int, int]:
    """Prime factorization for n <= ~2e18."""
    if n in _factor_cache:
        return dict(_factor_cache[n])

    def rec(m: int, acc: Dict[int, int]) -> None:
        if m == 1:
            return
        if _is_probable_prime(m):
            acc[m] = acc.get(m, 0) + 1
            return
        d = _pollard_rho(m)
        rec(d, acc)
        rec(m // d, acc)

    acc: Dict[int, int] = {}
    rec(n, acc)
    # normalize (Pollard Rho recursion can produce repeated primes)
    fac: Dict[int, int] = {}
    for p, e in acc.items():
        fac[p] = fac.get(p, 0) + e

    _factor_cache[n] = fac
    return dict(fac)


def _divisors_from_factor(fac: Dict[int, int]) -> List[int]:
    divs = [1]
    for p, e in fac.items():
        nxt = []
        pe = 1
        for _ in range(e + 1):
            for d in divs:
                nxt.append(d * pe)
            pe *= p
        divs = nxt
    return divs


_odd_divs_cache: Dict[int, List[int]] = {}


def _odd_divisors(n: int) -> List[int]:
    """Sorted list of odd divisors of n (includes 1)."""
    if n in _odd_divs_cache:
        return _odd_divs_cache[n]
    fac = _factor(n)
    divs = _divisors_from_factor(fac)
    divs = [d for d in divs if d % 2 == 1]
    divs.sort()
    _odd_divs_cache[n] = divs
    return divs


# ---------- primes by residue class ----------
def _primes_by_mod4(target_mod: int, count: int) -> List[int]:
    """Generate 'count' primes p with p % 4 == target_mod."""
    assert target_mod in (1, 3)
    primes: List[int] = []

    def is_prime_trial(m: int) -> bool:
        if m < 2:
            return False
        if m % 2 == 0:
            return m == 2
        r = int(math.isqrt(m))
        f = 3
        while f <= r:
            if m % f == 0:
                return False
            f += 2
        return True

    x = 2
    while len(primes) < count:
        if x % 4 == target_mod and is_prime_trial(x):
            primes.append(x)
        x += 1
    return primes


# We only ever need ~40 primes of each type (worst-case factor 3^38 < 2e18).
_P1 = _primes_by_mod4(1, 80)  # 1 mod 4
_P3 = _primes_by_mod4(3, 80)  # 3 mod 4

_LOG2_2 = _log2_int(2)
_LOG2_P1 = [_log2_int(p) for p in _P1]
_LOG2_P3 = [_log2_int(p) for p in _P3]


# ---------- minimizing numbers without constructing huge integers ----------
# Representation of a number built from the first len(exps) primes of some list:
#   value = Π primes[i] ** exps[i]
# along with its log2(value) for safe ordering.
Rep = Tuple[decimal.Decimal, Tuple[int, ...]]  # (log2, exponents)


_min_rep_cache: Dict[Tuple[int, int], Rep] = {}


def _min_rep_for_product(P: int, primes: List[int], logs: List[decimal.Decimal]) -> Rep:
    """
    Minimal number (in Rep form) such that:
        Π (2*e_i + 1) == P
    where e_i are the exponents of primes[0], primes[1], ...
    and exponents are non-increasing (standard "minimal by prime assignment" rule).

    P must be odd and >= 1.
    """
    if P == 1:
        return (decimal.Decimal(0), ())
    key = (P, primes[0])  # distinguish by prime set
    if key in _min_rep_cache:
        return _min_rep_cache[key]

    @functools.lru_cache(maxsize=None)
    def dfs(rem: int, idx: int, prev_f: int) -> Rep | None:
        if rem == 1:
            return (decimal.Decimal(0), ())
        if idx >= len(primes):
            return None

        best: Rep | None = None
        best_log: decimal.Decimal | None = None

        # choose a factor f = 2e+1 of rem, with f <= prev_f to keep e non-increasing
        for f in reversed(_odd_divisors(rem)):
            if f == 1 or f > prev_f:
                continue
            e = (f - 1) // 2
            if e <= 0:
                continue
            sub = dfs(rem // f, idx + 1, f)
            if sub is None:
                continue
            sub_log, sub_exps = sub
            cur_log = logs[idx] * e + sub_log
            if best is None or cur_log < best_log:
                best_log = cur_log
                best = (cur_log, (e,) + sub_exps)
        return best

    rep = dfs(P, 0, P)
    assert rep is not None
    _min_rep_cache[key] = rep
    return rep


def _rep_mod(rep: Rep, primes: List[int], mod: int) -> int:
    log2v, exps = rep
    r = 1
    for p, e in zip(primes, exps):
        r = (r * pow(p, e, mod)) % mod
    return r


def _rep_to_int(rep: Rep, primes: List[int]) -> int:
    """Only used for small test cases."""
    log2v, exps = rep
    x = 1
    for p, e in zip(primes, exps):
        x *= pow(p, e)
    return x


# For D = a2 * C, choose a2 (odd) to represent the contribution of 2's exponent,
# and C for primes p ≡ 3 (mod 4).
_best_D_cache: Dict[int, Tuple[decimal.Decimal, int, Rep]] = {}


def _best_rep_for_D(D: int) -> Tuple[decimal.Decimal, int, Rep]:
    """Return (log2, e2, rep3) minimizing 2^e2 * n3(C) under a2*C = D."""
    if D == 1:
        return (decimal.Decimal(0), 0, (decimal.Decimal(0), ()))
    if D in _best_D_cache:
        return _best_D_cache[D]

    best_log: decimal.Decimal | None = None
    best_e2 = 0
    best_rep3: Rep = (decimal.Decimal(0), ())

    for a2 in _odd_divisors(D):
        if a2 == 1:
            e2 = 0
            log2_part = decimal.Decimal(0)
        else:
            # a2 == 2*(e2-1)+1  =>  e2 == (a2+1)/2   (and e2>=2 here)
            e2 = (a2 + 1) // 2
            log2_part = _LOG2_2 * e2

        C = D // a2
        rep3 = _min_rep_for_product(C, _P3, _LOG2_P3)
        cur_log = log2_part + rep3[0]

        if best_log is None or cur_log < best_log:
            best_log = cur_log
            best_e2 = e2
            best_rep3 = rep3

    assert best_log is not None
    ans = (best_log, best_e2, best_rep3)
    _best_D_cache[D] = ans
    return ans


# Q representation bundle
_Q_cache: Dict[int, Tuple[decimal.Decimal, Rep, int, Rep]] = {}


def Q_rep(n: int) -> Tuple[decimal.Decimal, Rep, int, Rep]:
    """
    Return representation of Q(n) as (log2, rep1, e2, rep3):
      Q(n) = value(rep1 over primes 1 mod 4) * 2^e2 * value(rep3 over primes 3 mod 4)
    """
    if n in _Q_cache:
        return _Q_cache[n]

    # Key identity:
    # Let B = Π_{p≡1 mod4}(2e_p+1), A = d(m^2) = Π_{all primes}(2e'_p+1).
    # Then the total number of triples containing x is T(x)=(A+B-2)/2.
    # Setting T(x)=n gives A+B = 2(n+1).
    #
    # Since B is odd, B | (n+1). Enumerate B over odd divisors of (n+1).
    S = n + 1

    best_log: decimal.Decimal | None = None
    best_rep1: Rep = (decimal.Decimal(0), ())
    best_e2 = 0
    best_rep3: Rep = (decimal.Decimal(0), ())

    for B in _odd_divisors(S):
        rep1 = _min_rep_for_product(B, _P1, _LOG2_P1)
        D = (2 * S) // B - 1  # D = a2 * C
        logD, e2, rep3 = _best_rep_for_D(D)

        cur_log = rep1[0] + logD
        if best_log is None or cur_log < best_log:
            best_log = cur_log
            best_rep1 = rep1
            best_e2 = e2
            best_rep3 = rep3

    assert best_log is not None
    ans = (best_log, best_rep1, best_e2, best_rep3)
    _Q_cache[n] = ans
    return ans


def Q_mod(n: int, mod: int = MOD) -> int:
    log2v, rep1, e2, rep3 = Q_rep(n)
    r = _rep_mod(rep1, _P1, mod)
    r = (r * pow(2, e2, mod)) % mod
    r = (r * _rep_mod(rep3, _P3, mod)) % mod
    return r


def Q_exact(n: int) -> int:
    """Exact Q(n), intended only for small n (tests)."""
    log2v, rep1, e2, rep3 = Q_rep(n)
    return _rep_to_int(rep1, _P1) * (1 << e2) * _rep_to_int(rep3, _P3)


def solve() -> int:
    # Fixed seed for stable Pollard-Rho behavior
    random.seed(0)

    # Tests from the problem statement
    assert Q_exact(5) == 15
    assert Q_exact(10) == 48
    assert Q_exact(10**3) == 8064000

    total = 0
    for k in range(1, 19):
        total = (total + Q_mod(10**k, MOD)) % MOD
    return total


def main() -> None:
    print(solve())


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()
