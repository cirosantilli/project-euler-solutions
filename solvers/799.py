#!/usr/bin/env python3
"""
Project Euler 799 - Pentagonal Puzzle

Compute the smallest pentagonal number that can be written as a sum of two
pentagonal numbers in more than 100 different ways.

No external libraries are used.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple


# ----------------------------
# Pentagonal helpers
# ----------------------------


def pentagonal(n: int) -> int:
    return n * (3 * n - 1) // 2


# ----------------------------
# 64-bit primality + factoring
# (deterministic Miller-Rabin + Pollard Rho)
# ----------------------------

_MR_BASES_64 = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)


def _is_probable_prime(n: int) -> bool:
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small:
        if n % p == 0:
            return n == p

    # write n-1 = d * 2^s with d odd
    d = n - 1
    s = 0
    while d % 2 == 0:
        s += 1
        d //= 2

    for a in _MR_BASES_64:
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

    while True:
        c = random.randrange(1, n - 1)
        x = random.randrange(0, n)
        y = x
        d = 1

        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = math.gcd(abs(x - y), n)

        if d != n:
            return d


def factorize(n: int) -> Dict[int, int]:
    """Return prime factorization of n as {prime: exponent}."""
    if n <= 1:
        return {}

    stack = [n]
    primes: List[int] = []

    while stack:
        x = stack.pop()
        if x == 1:
            continue
        if _is_probable_prime(x):
            primes.append(x)
            continue
        d = _pollard_rho(x)
        stack.append(d)
        stack.append(x // d)

    primes.sort()
    fac: Dict[int, int] = {}
    for p in primes:
        fac[p] = fac.get(p, 0) + 1
    return fac


# ----------------------------
# Sum-of-two-squares machinery
# ----------------------------


def _sqrt_minus_one_mod_prime(p: int) -> int:
    """Find s such that s^2 ≡ -1 (mod p), for prime p ≡ 1 (mod 4)."""
    # Need a quadratic non-residue 'a'; then a^((p-1)/4) is sqrt(-1)
    for a in (2, 3, 5, 6, 7, 10, 11, 13, 17, 19, 23, 29):
        if a % p == 0:
            continue
        if pow(a, (p - 1) // 2, p) == p - 1:  # non-residue
            return pow(a, (p - 1) // 4, p)
    a = 2
    while True:
        if pow(a, (p - 1) // 2, p) == p - 1:
            return pow(a, (p - 1) // 4, p)
        a += 1


_cornacchia_cache: Dict[int, Tuple[int, int]] = {}
_sqrtm1_cache: Dict[int, int] = {}


def _cornacchia_prime_sum_squares(p: int) -> Tuple[int, int]:
    """
    For prime p ≡ 1 (mod 4), find (a,b) with a^2 + b^2 = p.
    Cornacchia's algorithm using a sqrt(-1) modulo p.
    """
    if p in _cornacchia_cache:
        return _cornacchia_cache[p]

    if p in _sqrtm1_cache:
        t = _sqrtm1_cache[p]
    else:
        t = _sqrt_minus_one_mod_prime(p)
        _sqrtm1_cache[p] = t

    def run(t0: int) -> Tuple[int, int]:
        r0, r1 = p, t0
        while r1 * r1 > p:
            r0, r1 = r1, r0 % r1
        a = r1
        b2 = p - a * a
        b = math.isqrt(b2)
        if b * b != b2:
            raise ValueError("Cornacchia failed")
        return a, b

    try:
        a, b = run(t)
    except ValueError:
        a, b = run(p - t)

    _cornacchia_cache[p] = (a, b)
    return a, b


def _gauss_mul(u: int, v: int, a: int, b: int) -> Tuple[int, int]:
    """(u+vi)*(a+bi)"""
    return u * a - v * b, u * b + v * a


def count_pentagonal_sum_ways(m: int) -> int:
    """
    Count ways to write P_m as sum of two pentagonal numbers.

    Using:
      24*P_n + 1 = (6n-1)^2
    If P_m = P_a + P_b, then:
      (6m-1)^2 + 1 = (6a-1)^2 + (6b-1)^2
    So we count representations of N = (6m-1)^2 + 1 as a sum of two squares
    with both squares coming from numbers ≡ 5 (mod 6), i.e. ≡ 2 (mod 3).
    """
    x = 6 * m - 1
    N = x * x + 1

    fac = factorize(N)

    # Build Gaussian integer factors for odd primes (all should be 1 mod 4)
    options: List[List[Tuple[int, int]]] = []
    for p, e in fac.items():
        if p == 2:
            continue
        a, b = _cornacchia_prime_sum_squares(p)
        gp = (a, b)
        gc = (a, -b)

        # powers gp^k and gc^k
        pow_gp: List[Tuple[int, int]] = [(1, 0)]
        for _ in range(e):
            uu, vv = pow_gp[-1]
            pow_gp.append(_gauss_mul(uu, vv, gp[0], gp[1]))

        pow_gc: List[Tuple[int, int]] = [(1, 0)]
        for _ in range(e):
            uu, vv = pow_gc[-1]
            pow_gc.append(_gauss_mul(uu, vv, gc[0], gc[1]))

        # (gp^k)*(gc^(e-k)) for k=0..e
        opt: List[Tuple[int, int]] = []
        for k in range(e + 1):
            u1, v1 = pow_gp[k]
            u2, v2 = pow_gc[e - k]
            opt.append(_gauss_mul(u1, v1, u2, v2))
        options.append(opt)

    reps: List[Tuple[int, int]] = [(1, 0)]
    for opt in options:
        new: List[Tuple[int, int]] = []
        for u, v in reps:
            for a, b in opt:
                new.append(_gauss_mul(u, v, a, b))
        reps = new

    # Multiply by factor 2 = (1+i)(1-i). Any one of them suffices; units cover signs.
    reps = [_gauss_mul(u, v, 1, 1) for (u, v) in reps]

    # Filter to u,v corresponding to 6n-1 ≡ 5 (mod 6) ⇔ u≡v≡2 (mod 3).
    seen = set()
    for u, v in reps:
        u = abs(u)
        v = abs(v)
        if u == 0 or v == 0:
            continue
        if u > v:
            u, v = v, u
        if (u % 3 == 2) and (v % 3 == 2):
            seen.add((u, v))
    return len(seen)


# ----------------------------
# Segmented sieve to find the first m with >100 ways
# ----------------------------


def _primes_up_to(n: int) -> List[int]:
    sieve = bytearray(b"\x01") * (n + 1)
    sieve[:2] = b"\x00\x00"
    r = int(n**0.5)
    for i in range(2, r + 1):
        if sieve[i]:
            step = i
            start = i * i
            sieve[start : n + 1 : step] = b"\x00" * (((n - start) // step) + 1)
    return [i for i in range(n + 1) if sieve[i]]


def _next_prime_after(n: int) -> int:
    x = n + 1
    if x % 2 == 0:
        x += 1
    while True:
        if _is_probable_prime(x):
            return x
        x += 2


def _precompute_roots(limit: int) -> Tuple[List[Tuple[int, int, int]], int]:
    """
    For primes p≡1 (mod4), precompute the two residues r such that:
      p | ((6m-1)^2 + 1)
    This yields m ≡ (1 ± s) / 6 (mod p) where s^2 ≡ -1 (mod p).
    """
    primes = _primes_up_to(limit)
    roots: List[Tuple[int, int, int]] = []
    for p in primes:
        if p % 4 != 1:
            continue
        s = _sqrt_minus_one_mod_prime(p)
        inv6 = pow(6, p - 2, p)  # modular inverse since p is prime
        r1 = ((1 + s) * inv6) % p
        r2 = ((1 - s) * inv6) % p
        roots.append((p, r1, r2))
    return roots, _next_prime_after(limit)


def _upper_factor_multiplier(rem: int, min_prime: int) -> int:
    """
    Upper bound on multiplicative factor contributed to Π(e+1) by the unknown part 'rem',
    assuming its prime factors are all >= min_prime and exponents are 1.
    """
    if rem <= 1:
        return 1
    # After sieving out all primes < min_prime, any remaining factor must be >= min_prime.
    # Max number of distinct such primes is floor(log_{min_prime}(rem)).
    k = int(math.log(rem, min_prime))  # k >= 1 if rem >= min_prime
    return 1 << k  # 2^k


def find_answer(block_size: int = 50_000, prime_limit: int = 200_000) -> int:
    roots, next_p = _precompute_roots(prime_limit)

    m0 = 1
    while True:
        B = block_size

        # residuals for odd part: ((6m-1)^2 + 1) // 2
        res = [0] * B
        prod = [1] * B

        x = 6 * m0 - 1
        n = x * x + 1
        for i in range(B):
            res[i] = n // 2
            x += 6
            n = x * x + 1

        # divide out small primes using precomputed roots
        m0_mod_cache = None  # computed lazily per prime (since m0 changes each block)
        for p, r1, r2 in roots:
            if m0_mod_cache is None:
                m0_mod_cache = {}
            # compute m0 mod p
            mp = m0_mod_cache.get(p)
            if mp is None:
                mp = m0 % p
                m0_mod_cache[p] = mp

            for r in (r1, r2):
                idx = (r - mp) % p
                while idx < B:
                    val = res[idx]
                    if val % p == 0:
                        e = 0
                        while val % p == 0:
                            val //= p
                            e += 1
                        res[idx] = val
                        prod[idx] *= e + 1
                    idx += p

        # check candidates in increasing m
        for i in range(B):
            pr = prod[i]
            rem = res[i]
            # Upper bound whether Π(e+1) could reach 202 (needed for >100 unordered reps).
            ub = pr * _upper_factor_multiplier(rem, next_p)
            if ub < 202:
                continue

            m = m0 + i
            # Exact check (a few candidates)
            ways = count_pentagonal_sum_ways(m)
            if ways > 100:
                return pentagonal(m)

        m0 += B


def main() -> None:
    # deterministic behavior
    random.seed(0)

    # Asserts for the sample values in the problem statement:
    # P_8 = 92 = P_4 + P_7
    assert count_pentagonal_sum_ways(8) == 1
    # P_49 = 3577 has two representations
    assert count_pentagonal_sum_ways(49) == 2
    # P_268 = 107602 has three representations
    assert count_pentagonal_sum_ways(268) == 3

    ans = find_answer()
    print(ans)


if __name__ == "__main__":
    main()
