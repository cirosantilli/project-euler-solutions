#!/usr/bin/env python3
"""
Project Euler 699: Triffle Numbers

We need T(N) = sum of all n <= N such that, after reducing sigma(n)/n to lowest terms a/b,
the denominator b is a positive power of 3.

Key idea:
- sigma is multiplicative.
- If we build n by multiplying by a new prime power p^e (p not dividing current n), then
  sigma(n)/n gets multiplied by sigma(p^e)/p^e.
- Because p never divides sigma(p^e), any new p^e appearing in the denominator can only be
  cancelled if p^e already divides the current numerator.
- Starting from n made only of 2, 3, 5 (i.e. 2^a * 3^b * 5^c), the reduced denominator is a
  divisor of that number, so it has no primes beyond {2,3,5}.
  If we only add primes that already divide the current numerator, the denominator never
  gains new prime factors, it can only shrink via gcd cancellations.
This makes a depth-first search feasible up to N = 10^14.
"""

from __future__ import annotations

import math
import random
import sys
from typing import Dict, List, Tuple


# -----------------------------
# 64-bit primality & factoring
# -----------------------------


def _is_probable_prime(n: int) -> bool:
    """Deterministic Miller–Rabin for 64-bit integers."""
    if n < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small_primes:
        if n % p == 0:
            return n == p

    # write n-1 = d * 2^s with d odd
    d = n - 1
    s = 0
    while (d & 1) == 0:
        s += 1
        d >>= 1

    # Deterministic bases for testing 64-bit integers
    # (well-known set; works for n < 2^64)
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
    """Return a non-trivial factor of n (n is composite, odd)."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3

    # Retry with different constants until we find a factor.
    while True:
        c = random.randrange(1, n)
        x = random.randrange(0, n)
        y = x
        d = 1

        # polynomial: x^2 + c (mod n)
        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = math.gcd(abs(x - y), n)
        if d != n:
            return d


def _factorize(n: int, out: Dict[int, int]) -> None:
    """Populate out with prime factorization of n."""
    if n == 1:
        return

    # Quick trial division by a few small primes helps Pollard Rho a lot.
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            cnt = 0
            while n % p == 0:
                n //= p
                cnt += 1
            out[p] = out.get(p, 0) + cnt
            if n == 1:
                return
            break

    if n == 1:
        return
    if _is_probable_prime(n):
        out[n] = out.get(n, 0) + 1
        return

    d = _pollard_rho(n)
    _factorize(d, out)
    _factorize(n // d, out)


def _factor_multiset(n: int) -> Dict[int, int]:
    res: Dict[int, int] = {}
    _factorize(n, res)
    return res


# -----------------------------
# Problem-specific helpers
# -----------------------------


def _is_power_of_3(n: int) -> int:
    """Return k if n == 3^k, else -1."""
    if n <= 0:
        return -1
    k = 0
    while n % 3 == 0:
        n //= 3
        k += 1
    return k if n == 1 else -1


def _sigma_prime_power(p: int, e: int) -> int:
    """sigma(p^e) = 1 + p + ... + p^e"""
    # geometric series: (p^(e+1)-1)/(p-1)
    return (pow(p, e + 1) - 1) // (p - 1)


def _seed_states(N: int) -> List[Tuple[int, int, int]]:
    """
    Enumerate all seeds n = 2^a * 3^b * 5^c <= N with b >= 1.

    For each seed, compute reduced fraction sigma(n)/n = num/den.
    Return (n, num, den).
    """
    pow2 = [1]
    while pow2[-1] * 2 <= N:
        pow2.append(pow2[-1] * 2)
    pow3 = [1]
    while pow3[-1] * 3 <= N:
        pow3.append(pow3[-1] * 3)
    pow5 = [1]
    while pow5[-1] * 5 <= N:
        pow5.append(pow5[-1] * 5)

    seeds: List[Tuple[int, int, int]] = []
    for a, pa in enumerate(pow2):
        sig2 = (pa * 2 - 1) if a > 0 else 1  # sigma(2^a)
        for b in range(1, len(pow3)):  # b>=1 so n divisible by 3
            pb = pow3[b]
            if pa * pb > N:
                break
            sig3 = (pb * 3 - 1) // 2  # sigma(3^b) = (3^(b+1)-1)/2
            for c, pc in enumerate(pow5):
                n = pa * pb * pc
                if n > N:
                    break
                sig5 = (pc * 5 - 1) // 4 if c > 0 else 1  # sigma(5^c)
                sig = sig2 * sig3 * sig5

                g = math.gcd(n, sig)
                num = sig // g
                den = n // g

                # If den has lost all factors of 3, it can never become a 3-power later
                # (den only shrinks in the DFS).
                if den % 3 != 0:
                    continue
                # If den==1, we can never reach k>0 (den doesn't grow).
                if den == 1:
                    continue

                seeds.append((n, num, den))
    return seeds


def triffle_sum(N: int) -> int:
    """
    Compute T(N): sum of all n <= N where sigma(n)/n reduces to a/b with b=3^k, k>0.
    """
    random.seed(0)
    sys.setrecursionlimit(2_000_000)

    # Factorization cache for numerators encountered.
    fac_cache: Dict[int, Dict[int, int]] = {}

    def factors_of(x: int) -> Dict[int, int]:
        if x in fac_cache:
            return fac_cache[x]
        d = _factor_multiset(x)
        fac_cache[x] = d
        return d

    seeds = _seed_states(N)

    visited = set()
    total = 0

    def dfs(n: int, num: int, den: int) -> None:
        nonlocal total
        if n in visited:
            return
        visited.add(n)

        k = _is_power_of_3(den)
        if k > 0:
            total += n

        # Denominator never grows; if it's already 1 or not divisible by 3,
        # it can never become 3^k with k>0.
        if den == 1 or den % 3 != 0:
            return

        if num == 1:
            return

        fac = factors_of(num)
        for p, exp in fac.items():
            if p <= 5:
                continue  # handled by seeds
            if n % p == 0:
                continue  # do not reuse a prime (keeps updates multiplicative)

            pp = 1
            for e in range(1, exp + 1):
                pp *= p
                if n > N // pp:
                    break

                # Multiply by p^e, cancelling p^e immediately.
                new_num = (num // pp) * _sigma_prime_power(p, e)
                new_den = den  # p^e cancels out completely

                g = math.gcd(new_num, new_den)
                new_num //= g
                new_den //= g
                dfs(n * pp, new_num, new_den)

    for n, num, den in seeds:
        dfs(n, num, den)

    return total


def main() -> None:
    # Test values from the statement
    assert triffle_sum(100) == 270
    assert triffle_sum(10**6) == 26089287

    print(triffle_sum(10**14))


if __name__ == "__main__":
    main()
