#!/usr/bin/env python3
"""
Project Euler 302 - Strong Achilles Numbers

Count strong Achilles numbers S < 10^18 where both S and φ(S) are Achilles.
An Achilles number is powerful (every prime exponent >= 2) but not a perfect power
(gcd of prime exponents equals 1).

This program uses a depth-first search over prime powers, tracking the prime
exponent multiset of φ(n) incrementally for pruning.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from typing import Dict, List, Tuple, Set


def icbrt(n: int) -> int:
    """Integer cube root: floor(cuberoot(n)) for n >= 0."""
    if n <= 0:
        return 0
    lo, hi = 0, 1
    while hi * hi * hi <= n:
        hi *= 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if mid * mid * mid <= n:
            lo = mid
        else:
            hi = mid
    return lo


def linear_sieve_spf(n: int) -> Tuple[List[int], List[int]]:
    """
    Linear sieve producing primes up to n and smallest prime factor (spf) for each number.
    spf[x] == 0 for x < 2.
    """
    spf = [0] * (n + 1)
    primes: List[int] = []
    for i in range(2, n + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
        for p in primes:
            v = i * p
            if v > n or p > spf[i]:
                break
            spf[v] = p
    return primes, spf


def factorize_with_spf(x: int, spf: List[int]) -> List[Tuple[int, int]]:
    """Prime factorization as (prime, exponent) pairs for x >= 1."""
    res: List[Tuple[int, int]] = []
    while x > 1:
        p = spf[x]
        cnt = 1
        x //= p
        while x % p == 0:
            cnt += 1
            x //= p
        res.append((p, cnt))
    return res


def count_strong_achilles(upper_exclusive: int) -> int:
    """
    Return the count of strong Achilles numbers S with 0 < S < upper_exclusive.
    """
    limit = upper_exclusive - 1  # inclusive maximum
    if limit < 1:
        return 0

    # Strong Achilles numbers have at least two distinct prime factors.
    # The largest prime factor must have exponent >= 3, and we need room for at least 2^2.
    max_p = icbrt(limit // 4)
    if max_p < 3:
        return 0

    primes, spf = linear_sieve_spf(max_p)

    # Precompute factorization of (p-1) for each prime p in primes.
    pm1: List[List[Tuple[int, int]]] = [None] * len(primes)  # type: ignore[assignment]
    for i, p in enumerate(primes):
        pm1[i] = factorize_with_spf(p - 1, spf) if p > 2 else []

    def phi_gcd_is_one(phi_f: Dict[int, int]) -> bool:
        """
        φ(n) is a perfect power iff gcd of its prime exponents > 1.
        At call sites we only call this when φ is already powerful (no exponent 1).
        """
        g = 0
        for e in phi_f.values():
            g = math.gcd(g, e)
            if g == 1:
                return True
        return False

    def apply_prime(
        idx: int,
        p: int,
        e: int,
        phi_f: Dict[int, int],
        ones: Set[int],
    ) -> List[Tuple[int, int, bool]]:
        """
        Multiply n by p^e and update factorization of φ(n):
          φ(p^e) = p^(e-1) * (p-1)
        The caller maintains n separately; this only updates phi_f and ones.

        Returns a change-list for backtracking: (prime, old_exp, existed_before).
        """
        changes: List[Tuple[int, int, bool]] = []

        # Contribution of p^(e-1)
        existed = p in phi_f
        old = phi_f[p] if existed else 0
        changes.append((p, old, existed))
        new = old + (e - 1)
        phi_f[p] = new
        if old == 1:
            ones.discard(p)
        if new == 1:
            ones.add(p)

        # Contribution of (p-1)
        for q, a in pm1[idx]:
            existed = q in phi_f
            old = phi_f[q] if existed else 0
            changes.append((q, old, existed))
            new = old + a
            phi_f[q] = new
            if old == 1:
                ones.discard(q)
            if new == 1:
                ones.add(q)

        return changes

    def undo(
        changes: List[Tuple[int, int, bool]],
        phi_f: Dict[int, int],
        ones: Set[int],
    ) -> None:
        """Undo apply_prime using its returned change-list."""
        for q, old, existed in reversed(changes):
            if existed:
                phi_f[q] = old
            else:
                del phi_f[q]
            if old == 1:
                ones.add(q)
            else:
                ones.discard(q)

    def dfs(
        max_idx: int,
        n: int,
        gcd_n: int,
        phi_f: Dict[int, int],
        ones: Set[int],
        num_primes: int,
    ) -> int:
        """
        Explore adding primes strictly below primes[max_idx] (by index) to n.

        - gcd_n: gcd of exponents in n so far (gcd_n == 1 => not a perfect power)
        - phi_f: prime->exponent map for φ(n) so far
        - ones: primes whose exponent in φ(n) is exactly 1 (must be eliminated eventually)
        """
        # Need at least two distinct primes in n.
        if num_primes == 1 and n * 4 > limit:
            return 0

        # If φ has a prime with exponent 1, we cannot go below the largest such prime:
        # smaller primes r can't contribute that large prime via (r-1).
        if ones:
            qmax = max(ones)
            if n * qmax * qmax > limit:
                return 0
            lb = qmax
        else:
            lb = 2

        cnt = 0

        # Count n itself if it's already a strong Achilles number.
        if not ones and num_primes >= 2 and gcd_n == 1:
            if phi_gcd_is_one(phi_f):
                cnt += 1

        if max_idx < 0:
            return cnt

        rem = limit // n
        if rem < 4:
            return cnt

        # Next prime factor p must satisfy p^2 <= rem (since exponents in n are >=2).
        max_p2 = math.isqrt(rem)
        upper_val = primes[max_idx]
        if max_p2 < upper_val:
            upper_val = max_p2
        if upper_val < lb:
            return cnt

        upper_idx = bisect_right(primes, upper_val) - 1
        if upper_idx > max_idx:
            upper_idx = max_idx
        lb_idx = bisect_left(primes, lb)
        if lb_idx > upper_idx:
            return cnt

        for i in range(upper_idx, lb_idx - 1, -1):
            p = primes[i]

            # If p is already present in φ(n) (possibly with exponent 1), we may use exponent 2 in n.
            # Otherwise exponent 2 would create p^1 in φ(n), which is impossible to fix later.
            present_in_phi = p in phi_f
            min_e = 2 if present_in_phi else 3

            p_pow = p * p if min_e == 2 else p * p * p
            if p_pow > rem:
                continue

            e = min_e
            while p_pow <= rem:
                changes = apply_prime(i, p, e, phi_f, ones)
                cnt += dfs(
                    i - 1, n * p_pow, math.gcd(gcd_n, e), phi_f, ones, num_primes + 1
                )
                undo(changes, phi_f, ones)

                p_pow *= p
                e += 1

        return cnt

    total = 0

    # Choose the largest prime factor. Its exponent must be >= 3.
    for idx in range(len(primes) - 1, 0, -1):  # idx > 0 ensures largest prime >= 3
        p = primes[idx]
        p_pow = p * p * p
        if p_pow * 4 > limit:
            continue

        e = 3
        while p_pow * 4 <= limit:
            n = p_pow
            phi_f: Dict[int, int] = {p: e - 1}  # contributes p^(e-1) to φ(n)
            ones: Set[int] = set()

            # Multiply φ(n) by (p-1)
            for q, a in pm1[idx]:
                old = phi_f.get(q, 0)
                new = old + a
                phi_f[q] = new
                if old == 1:
                    ones.discard(q)
                if new == 1:
                    ones.add(q)

            total += dfs(idx - 1, n, e, phi_f, ones, 1)

            p_pow *= p
            e += 1

    return total


def main() -> None:
    # Tests from the problem statement
    assert count_strong_achilles(10_000) == 7
    assert count_strong_achilles(100_000_000) == 656

    print(count_strong_achilles(10**18))


if __name__ == "__main__":
    main()
