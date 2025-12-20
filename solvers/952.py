#!/usr/bin/env python3
"""Project Euler 952: Order Modulo Factorial

Compute R(p, n): the multiplicative order of prime p modulo n! (with n < p).
For the actual Euler query we output R(10^9+7, 10^7) modulo 10^9+7.

No third-party libraries are used.
"""

from __future__ import annotations

import sys
from array import array


def linear_sieve_spf(limit: int) -> tuple[list[int], array]:
    """Return (primes, spf) up to limit inclusive using a linear sieve.

    spf[x] is the smallest prime factor of x for x>=2 (0 for 0/1).
    """
    spf = array("I", [0]) * (limit + 1)
    primes: list[int] = []

    spf_local = spf
    primes_append = primes.append

    for i in range(2, limit + 1):
        if spf_local[i] == 0:
            spf_local[i] = i
            primes_append(i)
        for p in primes:
            ip = i * p
            if ip > limit:
                break
            spf_local[ip] = p
            if p == spf_local[i]:
                break

    return primes, spf


def v_p_factorial(n: int, p: int) -> int:
    """Exponent of prime p in n! (Legendre)."""
    s = 0
    while n:
        n //= p
        s += n
    return s


def v2(x: int) -> int:
    """2-adic valuation for positive x."""
    c = 0
    while (x & 1) == 0:
        x >>= 1
        c += 1
    return c


def order_mod_2_power_exponent(a: int, k: int) -> int:
    """Return t such that ord_{2^k}(a) = 2^t for odd a and k>=1."""
    # Mod 2: everything odd is 1
    if k <= 1:
        return 0
    # Mod 4: order is 1 if a≡1, else 2
    if k == 2:
        return 0 if (a & 3) == 1 else 1

    # k >= 3
    if (a & 3) == 1:
        # LTE: v2(a^{2^t}-1) = v2(a-1) + t for t>=0
        t = k - v2(a - 1)
        return t if t > 0 else 0
    else:
        # a ≡ 3 (mod 4): need t>=1 and
        # v2(a^{2^t}-1) = v2(a-1)+v2(a+1)+t-1 for t>=1
        t = k - v2(a + 1)
        return t if t > 1 else 1


def factorize_spf(x: int, spf: array) -> list[tuple[int, int]]:
    """Prime factorization of x using smallest prime factors."""
    res: list[tuple[int, int]] = []
    while x > 1:
        p = spf[x]
        e = 0
        while x % p == 0:
            x //= p
            e += 1
        res.append((p, e))
    return res


def multiplicative_order_mod_prime_and_update_lcm(
    p: int,
    q: int,
    spf: array,
    max_exp: array,
) -> int:
    """Compute ord_q(p) (q odd prime) while updating max_exp with its factorization.

    Updates max_exp for the prime factors of ord_q(p).
    Returns r0 = ord_q(p).
    """
    # Factor q-1
    m = q - 1
    factors = factorize_spf(m, spf)

    r = m
    base = p % q
    pow_ = pow

    # Reduce r by prime factors of q-1
    for f, e in factors:
        e_rem = e
        while e_rem:
            cand = r // f
            if pow_(base, cand, q) == 1:
                r = cand
                e_rem -= 1
            else:
                break
        # Whatever exponent remains contributes to ord_q(p)
        if e_rem > max_exp[f]:
            max_exp[f] = e_rem

    return r


def q_adic_valuation_of_p_pow_r_minus_1(p: int, r: int, q: int, limit: int) -> int:
    """Compute s = v_q(p^r - 1), but never exceeding `limit`.

    For odd q, s is typically very small; we find it by testing modulo q^k.
    """
    if limit <= 1:
        return 1

    pow_ = pow
    s = 1
    q_pow = q
    while s < limit:
        q_pow_next = q_pow * q
        if pow_(p, r, q_pow_next) != 1:
            break
        q_pow = q_pow_next
        s += 1
    return s


def R_order_mod_factorial(p: int, n: int, mod: int | None = None) -> int:
    """Return R(p,n), the multiplicative order of p modulo n!.

    If mod is provided, returns R(p,n) modulo mod.
    """
    if n <= 1:
        return 1 if mod is None else (1 % mod)

    primes, spf = linear_sieve_spf(n)
    max_exp = array("I", [0]) * (n + 1)

    # Handle q=2 separately (largest prime power of 2 dividing n!)
    a2 = v_p_factorial(n, 2)
    t2 = order_mod_2_power_exponent(p, a2)
    if t2 > max_exp[2]:
        max_exp[2] = t2

    # Odd primes
    spf_local = spf
    max_exp_local = max_exp

    for q in primes:
        if q == 2:
            continue

        # a = exponent of q in n!
        a = 0
        nn = n
        while nn:
            nn //= q
            a += nn

        # r0 = ord_q(p) and update lcm exponents from r0's factorization
        r0 = multiplicative_order_mod_prime_and_update_lcm(
            p, q, spf_local, max_exp_local
        )

        # s = v_q(p^{r0} - 1), then order modulo q^a multiplies by q^{max(0, a-s)}
        s = q_adic_valuation_of_p_pow_r_minus_1(p, r0, q, a)
        extra = a - s
        if extra > max_exp_local[q]:
            max_exp_local[q] = extra

    # Reconstruct order (exactly for small n) or compute modulo.
    if mod is None:
        res = 1
        for q in primes:
            e = max_exp_local[q]
            if e:
                res *= q**e
        return res

    res = 1 % mod
    for q in primes:
        e = max_exp_local[q]
        if e:
            res = (res * pow(q, e, mod)) % mod
    return res


def main() -> None:
    P = 10**9 + 7

    # Tests from the problem statement
    assert R_order_mod_factorial(7, 4) == 2
    assert R_order_mod_factorial(P, 12) == 17280

    # Euler query
    ans = R_order_mod_factorial(P, 10**7, mod=P)
    print(ans)


if __name__ == "__main__":
    sys.setrecursionlimit(1_000_000)
    main()
