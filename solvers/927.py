#!/usr/bin/env python3
"""Project Euler 927: Prime-ary Tree

We have the recurrence
    t_k(0) = 1
    t_k(n) = 1 + t_k(n-1)^k   (n >= 1)

For a prime p, S_p is the set of m for which t_p(n) == 0 (mod m) for some n.
Let S = intersection over all primes p of S_p, and R(N) = sum of m in S, m<=N.

This program computes R(10^7) without any external libraries.
"""

from __future__ import annotations

import math
import sys


def primes_upto(n: int) -> list[int]:
    """Return all primes <= n using an odd-only sieve (bytearray)."""
    if n < 2:
        return []
    if n == 2:
        return [2]

    size = n // 2 + 1  # represents odds: value = 2*i+1
    is_prime = bytearray(b"\x01") * size
    is_prime[0] = 0  # 1 is not prime

    limit = int(n**0.5)
    for x in range(3, limit + 1, 2):
        if is_prime[x // 2]:
            start = (x * x) // 2
            step = x
            is_prime[start::step] = b"\x00" * (((size - 1 - start) // step) + 1)

    primes = [2]
    primes.extend(2 * i + 1 for i in range(1, size) if is_prime[i])
    return primes


def prime_factors_unique(n: int, primes_for_trial: list[int]) -> list[int]:
    """Return the distinct prime factors of n (n <= 10^7 here)."""
    res: list[int] = []
    x = n
    for p in primes_for_trial:
        if p * p > x:
            break
        if x % p == 0:
            res.append(p)
            while x % p == 0:
                x //= p
    if x > 1:
        res.append(x)
    return res


def hits_zero_mod_prime(q: int, k: int) -> bool:
    """Return True iff the orbit of 1 under x -> 1 + x^k (mod q) hits 0.

    q is prime.

    Key shortcut: If gcd(k, q-1) == 1, then x -> x^k permutes F_q, hence
    x -> 1 + x^k is a permutation of F_q. Since 0 maps to 1, 0 and 1 are in the
    same cycle, so starting from 1 we must eventually reach 0.

    For the remaining case k | (q-1) (k is prime in our usage), we detect a cycle
    using Brent's algorithm and watch for 0 along the orbit.
    """
    if q == 2:
        return True

    # If gcd(k, q-1) == 1 (equivalently (q-1) % k != 0 since k is prime), the map
    # is a permutation and 0 must be reached.
    if (q - 1) % k != 0:
        return True

    mod = q

    if k == 2:

        def f(x: int) -> int:
            return (x * x + 1) % mod

    else:
        pow_mod = pow

        def f(x: int) -> int:
            return (pow_mod(x, k, mod) + 1) % mod

    # Brent cycle detection, advancing hare one step per iteration.
    tortoise = 1
    hare = f(tortoise)
    if hare == 0:
        return True

    power = 1
    lam = 1
    while tortoise != hare:
        if hare == 0:
            return True
        if power == lam:
            tortoise = hare
            power <<= 1
            lam = 0
        hare = f(hare)
        lam += 1

    # If we exited because hare == tortoise and hare == 0, it's still a hit.
    return hare == 0


def good_primes_upto(limit: int) -> list[int]:
    """Return primes q <= limit that lie in S (i.e., in S_p for every prime p)."""
    all_primes = primes_upto(limit)
    trial_primes = primes_upto(int(math.isqrt(limit)) + 1)

    goods: list[int] = []
    for q in all_primes:
        if q == 2:
            goods.append(q)
            continue

        # For odd q, we must check exponent 2 because 2 | (q-1).
        # If q ≡ 3 (mod 4), -1 is not a quadratic residue and 0 is unreachable.
        if (q & 3) != 1:
            continue
        if not hits_zero_mod_prime(q, 2):
            continue

        # For primes p not dividing q-1, gcd(p, q-1)=1 => automatic success.
        # So only primes dividing (q-1) need to be tested.
        factors = prime_factors_unique(q - 1, trial_primes)
        ok = True
        for p in factors:
            if p == 2:
                continue
            if not hits_zero_mod_prime(q, p):
                ok = False
                break
        if ok:
            goods.append(q)

    return goods


def sum_squarefree_products_leq(limit: int, primes_list: list[int]) -> int:
    """Sum of all squarefree products of a subset of primes_list that are <= limit."""
    prods = [1]
    for p in primes_list:
        base_len = len(prods)
        for i in range(base_len):
            v = prods[i] * p
            if v <= limit:
                prods.append(v)
    return sum(prods)


def R(limit: int) -> int:
    goods = good_primes_upto(limit)
    return sum_squarefree_products_leq(limit, goods)


def main() -> None:
    # Tests given in the problem statement.
    assert R(20) == 18
    assert R(1000) == 2089

    n = 10_000_000
    if len(sys.argv) >= 2:
        n = int(sys.argv[1])
    print(R(n))


if __name__ == "__main__":
    main()
