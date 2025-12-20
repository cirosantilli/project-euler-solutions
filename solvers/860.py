#!/usr/bin/env python3
"""
Project Euler 860: Gold and Silver Coin Game

We count "fair" arrangements of n stacks of height 2, where the first player loses
whether the first mover is Gold (Gary) or Silver (Sally), assuming optimal play.

This solution uses combinatorial game theory (each stack is a number game), reducing
the problem to counting sequences whose game-values sum to 0, then computes the count
modulo 989898989.
"""

from __future__ import annotations


MOD = 989_898_989
N = 9898


def modinv(a: int, mod: int) -> int:
    """Multiplicative inverse of a modulo mod, assuming gcd(a, mod) == 1."""
    a %= mod
    if a == 0:
        raise ZeroDivisionError("0 has no inverse modulo mod")
    # Extended Euclidean algorithm (iterative)
    t0, t1 = 0, 1
    r0, r1 = mod, a
    while r1:
        q = r0 // r1
        r0, r1 = r1, r0 - q * r1
        t0, t1 = t1, t0 - q * t1
    if r0 != 1:
        raise ValueError("inverse does not exist (not coprime)")
    return t0 % mod


def prepare_factorials(n: int, mod: int) -> tuple[list[int], list[int]]:
    """Return factorials and inverse factorials mod 'mod' for 0..n."""
    fac = [1] * (n + 1)
    for i in range(1, n + 1):
        fac[i] = (fac[i - 1] * i) % mod

    inv_fac = [1] * (n + 1)
    inv_fac[n] = modinv(fac[n], mod)
    for i in range(n, 0, -1):
        inv_fac[i - 1] = (inv_fac[i] * i) % mod

    return fac, inv_fac


def count_fair_mod(n: int, mod: int) -> int:
    """
    Count fair arrangements of n stacks (height 2), modulo mod.

    Each stack is one of four types:
      GG -> value  +2
      SS -> value  -2
      GS -> value +1/2
      SG -> value -1/2

    Fair  <=> total game value == 0.
    Multiply by 2 to avoid fractions: steps are {+4, -4, +1, -1} and we count
    sequences of length n summing to 0.

    Let t be the number of ±4 steps. Let m be the net count (#+4 - #-4) among those.
    Then among the remaining s = n-t ±1 steps, the net must be -4m.
    The number of sequences for given (t, m) is:
      C(n, t) * C(t, (t+m)/2) * C(s, (s-4m)/2)
    summed over valid t and m.
    """
    fac, inv_fac = prepare_factorials(n, mod)

    # Local bindings for speed
    fac_l = fac
    inv_l = inv_fac
    mod_l = mod
    fac_n = fac_l[n]

    total = 0
    for t in range(n + 1):
        s = n - t
        if s & 1:
            continue  # s must be even
        # C(n, t)
        choose_nt = fac_n * inv_l[t] % mod_l * inv_l[s] % mod_l

        high = t if t < (s // 4) else (s // 4)
        parity = t & 1

        inner = 0
        fac_t = fac_l[t]
        fac_s = fac_l[s]

        # Range of m is symmetric; sum m >= 0 and double m>0
        for m in range(parity, high + 1, 2):
            k = (t + m) // 2  # #(+4) among ±4 steps
            r = (s - 4 * m) // 2  # #(+1) among ±1 steps

            ct = fac_t * inv_l[k] % mod_l * inv_l[t - k] % mod_l
            cs = fac_s * inv_l[r] % mod_l * inv_l[s - r] % mod_l
            term = (ct * cs) % mod_l

            inner += term if m == 0 else (term << 1)

        total = (total + choose_nt * (inner % mod_l)) % mod_l

    return total


def count_fair_exact(n: int) -> int:
    """Exact (integer) version for small n, used for statement checks."""
    from math import comb

    total = 0
    for t in range(n + 1):
        s = n - t
        if s & 1:
            continue
        choose_nt = comb(n, t)
        high = min(t, s // 4)
        parity = t & 1
        inner = 0
        for m in range(parity, high + 1, 2):
            k = (t + m) // 2
            r = (s - 4 * m) // 2
            term = comb(t, k) * comb(s, r)
            inner += term if m == 0 else 2 * term
        total += choose_nt * inner
    return total


def _self_test() -> None:
    # Test values given in the problem statement:
    assert count_fair_exact(2) == 4
    assert count_fair_exact(10) == 63_594

    # Cross-check modular path on the same small values
    assert count_fair_mod(2, MOD) == 4
    assert count_fair_mod(10, MOD) == 63_594


def main() -> None:
    _self_test()
    print(count_fair_mod(N, MOD))


if __name__ == "__main__":
    main()
