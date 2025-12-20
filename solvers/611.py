#!/usr/bin/env python3
"""
Project Euler 611 — Hallway of Square Steps

Peter toggles door n once for every pair of distinct squares a^2 < b^2 with a^2 + b^2 = n.
Thus door n is open iff the number of such representations is odd.

This solver follows the classic sum-of-two-squares factorization/parity analysis and
uses fast prime counting for primes p ≡ 1 (mod 4) up to large limits.
No external libraries are used (stdlib only).
"""

from __future__ import annotations

from array import array
from math import isqrt


# ----------------------------
# Small verification (bruteforce)
# ----------------------------


def F_bruteforce(N: int) -> int:
    """
    Directly simulate toggles by enumerating pairs a<b with a^2+b^2<=N.
    Fast enough up to N=10^6 (used only for asserts from the statement).
    """
    toggled = bytearray(N + 1)
    lim = isqrt(N)
    for a in range(1, lim + 1):
        a2 = a * a
        for b in range(a + 1, lim + 1):
            s = a2 + b * b
            if s > N:
                break
            toggled[s] ^= 1
    return int(sum(toggled))


# ----------------------------
# Sieve utilities (SPF + primes)
# ----------------------------


def sieve_spf(n: int) -> tuple[array, list[int]]:
    """
    Linear sieve producing smallest prime factor (spf) for every 0..n,
    and the list of primes <= n.
    """
    spf = array("I", [0]) * (n + 1)
    primes: list[int] = []
    for i in range(2, n + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
        for p in primes:
            ip = i * p
            if ip > n:
                break
            spf[ip] = p
            if p == spf[i]:
                break
    return spf, primes


def build_pi1_prefix(limit: int, primes: list[int]) -> array:
    """
    pi1[x] = number of primes <= x with p ≡ 1 (mod 4), for x <= limit.
    """
    is_p = bytearray(limit + 1)
    for p in primes:
        if p <= limit:
            is_p[p] = 1

    pi1 = array("I", [0]) * (limit + 1)
    c = 0
    for x in range(limit + 1):
        if is_p[x] and (x & 3) == 1:
            c += 1
        pi1[x] = c
    return pi1


# ----------------------------
# Prime counting table
#   We compute two tables for all distinct values v = floor(N / i):
#     pi(v)   = number of primes <= v
#     S(v)    = sum_{p<=v} chi(p), with chi the mod-4 character:
#               chi(p)= 0 if p=2, +1 if p≡1 (mod4), -1 if p≡3 (mod4)
#   Then pi1(v) = ( (pi(v) - 1_{v>=2}) + S(v) ) / 2.
# ----------------------------


def pi_and_char(
    N: int, primes: list[int], root: int
) -> tuple[array, array, array, array]:
    """
    Return (vals, idx_big, pi, schi) where:
      - vals: descending array of distinct values floor(N / i)
      - idx_big: array mapping t (= floor(N / v)) to index in vals, for v>root
      - pi[i] = π(vals[i])
      - schi[i] = Σ_{p<=vals[i]} chi(p)
    """
    # distinct quotients vals = floor(N/i)
    vals = array("q")
    i = 1
    while i <= N:
        q = N // i
        vals.append(q)
        i = N // q + 1

    m = len(vals)

    # idx_big[t] gives index where vals[index] = floor(N / t), for vals > root
    idx_big = array("I", [0]) * (root + 1)
    for idx, v in enumerate(vals):
        if v > root:
            idx_big[N // v] = idx
        else:
            break

    # initial g arrays:
    #  pi_init(v)   = count of integers in [2..v] = v-1
    #  schi_init(v) = sum_{n=2..v} chi(n)
    pi = array("q", [0]) * m
    schi = array("q", [0]) * m
    for j in range(m):
        v = vals[j]
        if v >= 2:
            pi[j] = v - 1
        # S_int(v) = sum_{n=1..v} chi(n) = count(1 mod4) - count(3 mod4)
        s_int = ((v + 3) // 4) - ((v + 1) // 4)
        schi[j] = s_int - 1  # exclude n=1

    # boundary: number of vals >= threshold (vals is descending)
    def upper_bound(threshold: int) -> int:
        lo, hi = 0, m
        while lo < hi:
            mid = (lo + hi) // 2
            if vals[mid] >= threshold:
                lo = mid + 1
            else:
                hi = mid
        return lo

    # localize for speed
    vals_l = vals
    pi_l = pi
    schi_l = schi
    idx_big_l = idx_big
    m_l = m
    root_l = root
    N_l = N

    for p in primes:
        p2 = p * p
        if p2 > N_l:
            break
        k = upper_bound(p2)
        if k == 0:
            break

        # index for (p-1) is always in the "small" tail since p<=root
        base_idx = m_l - (p - 1)
        base_pi = pi_l[base_idx]
        base_s = schi_l[base_idx]
        # chi(p)
        chi_p = 0 if p == 2 else (1 if (p & 3) == 1 else -1)

        for j in range(k):
            v = vals_l[j]
            vp = v // p
            if vp <= root_l:
                idx_vp = m_l - vp
            else:
                idx_vp = idx_big_l[N_l // vp]

            pi_l[j] -= pi_l[idx_vp] - base_pi
            if chi_p:
                schi_l[j] -= chi_p * (schi_l[idx_vp] - base_s)

    return vals, idx_big, pi, schi


# ----------------------------
# Factor helper for parity conditions
# ----------------------------


def parity_and_excluded_primes(u: int, spf: array) -> tuple[int, list[int]]:
    """
    For odd u:
      - parity: parity of the number of primes p ≡ 1 (mod4) with odd exponent in u.
      - excluded: sorted list of such primes (to exclude them as the special prime p in case B).
    """
    parity = 0
    excluded: list[int] = []
    while u > 1:
        p = spf[u]
        odd = 0
        while u % p == 0:
            u //= p
            odd ^= 1
        if odd and (p & 3) == 1:
            parity ^= 1
            excluded.append(p)
    return parity, excluded


# ----------------------------
# Main fast count
# ----------------------------


def F_fast(N: int) -> int:
    """
    Count doors left open after all actions for a given N.

    Door n is open iff the number of representations n = a^2 + b^2 with 0<a<b is odd.

    Using factorization of n and the sum-of-two-squares theorem, the parity condition splits into:
      (A) n = 2^k * u^2 with u odd, and u has an odd number of p≡1(mod4) to odd exponent.
      (B) n = 2^k * p * u^2 where p≡1(mod4) is prime and p has even exponent inside u
          (equivalently p not among the p≡1(mod4) primes with odd exponent in u).

    We enumerate odd u <= sqrt(N), factor u via SPF, and use fast π1(x) queries.
    """
    root = isqrt(N)
    spf, primes = sieve_spf(root)
    pi1_small = build_pi1_prefix(root, primes)

    # Build prime counting tables for large x (x>root). This is the heavy step.
    vals, idx_big, pi_tab, schi_tab = pi_and_char(N, primes, root)
    m = len(vals)

    def pi1_query(x: int) -> int:
        """π1(x) = number of primes <= x with p ≡ 1 (mod4)."""
        if x < 5:
            return 0
        if x <= root:
            return int(pi1_small[x])
        # x appears as a distinct floor(N / i) in our usage
        i = N // x
        idx = idx_big[i]
        pi_x = pi_tab[idx]  # π(x), includes prime 2 if x>=2
        s_x = schi_tab[idx]  # Σ chi(p) up to x (chi(2)=0)
        return int((pi_x - 1 + s_x) // 2)

    total = 0

    # Enumerate odd u; u^2 <= N
    for u in range(1, root + 1, 2):
        u2 = u * u
        max2 = N // u2  # == floor(N / u^2)

        parity, excluded = parity_and_excluded_primes(u, spf)

        # Case (A): n = 2^k * u^2 with parity condition (odd)
        if parity:
            total += max2.bit_length()  # count k>=0 with 2^k <= max2

        # Case (B): n = 2^k * p * u^2, p ≡ 1 (mod4), p not in excluded
        x = max2
        while x >= 5:
            cnt = pi1_query(x)
            if excluded:
                # excluded is sorted; count how many <= x
                ex = 0
                for p in excluded:
                    if p <= x:
                        ex += 1
                    else:
                        break
                cnt -= ex
            total += cnt
            x //= 2

    return total


def main() -> None:
    # Given checks from the problem statement
    assert F_bruteforce(5) == 1
    assert F_bruteforce(100) == 27
    assert F_bruteforce(1000) == 233
    assert F_bruteforce(10**6) == 112168

    # Solve the required instance
    print(F_fast(10**12))


if __name__ == "__main__":
    main()
