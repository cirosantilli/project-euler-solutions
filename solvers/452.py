#!/usr/bin/env python3
"""
Project Euler 452

Let N = 10^9 and MOD = 1234567891.

Define g(m, n) = number of n-tuples (a1..an) of positive integers with product exactly m.
For m = ∏ p^e, we can distribute each exponent e among n slots independently:
g(m, n) = ∏ C(n + e - 1, e).

The required count of n-tuples with product <= N is:
F(N, n) = ∑_{m=1..N} g(m, n)  (mod MOD).

For this problem n = N, so we compute:
Answer = ∑_{m<=N} ∏_{p^e || m} C(N + e - 1, e)  (mod MOD).

Key facts:
- Exponents e are small: for N=10^9, e <= floor(log2 N) = 29.
- The value on prime powers depends only on e, not on the prime itself.
- We can sum this multiplicative function using recursion over prime powers, and count
  the remaining "one large prime" cases with a fast prime counting function π(x).

No third-party libraries; single-threaded.
"""

from __future__ import annotations

import math
from functools import lru_cache

MOD = 1234567891
SIEVE_MAX = 10**6  # enough for Lehmer π(n) up to 10^9


def _sieve(n: int):
    """Return (primes, pi_prefix) for 0..n."""
    is_p = bytearray(b"\x01") * (n + 1)
    if n >= 0:
        is_p[0] = 0
    if n >= 1:
        is_p[1] = 0
    r = int(math.isqrt(n))
    for i in range(2, r + 1):
        if is_p[i]:
            step = i
            start = i * i
            is_p[start : n + 1 : step] = b"\x00" * (((n - start) // step) + 1)
    primes = [i for i in range(2, n + 1) if is_p[i]]
    pi = [0] * (n + 1)
    c = 0
    for i in range(n + 1):
        if is_p[i]:
            c += 1
        pi[i] = c
    return primes, pi


PRIMES, PI_SMALL = _sieve(SIEVE_MAX)


def _iroot(n: int, k: int) -> int:
    """Integer floor of n**(1/k), without relying on floating correctness."""
    if n < 2:
        return n
    lo, hi = 1, 1
    while hi**k <= n:
        hi *= 2
    lo = hi // 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if mid**k <= n:
            lo = mid
        else:
            hi = mid
    return lo


@lru_cache(maxsize=None)
def _phi(x: int, a: int) -> int:
    """Count of integers in [1..x] not divisible by the first a primes."""
    if a == 0:
        return x
    # small hard-coded shortcuts help a bit
    if a == 1:
        return x - x // 2
    if a == 2:
        return x - x // 2 - x // 3 + x // 6
    return _phi(x, a - 1) - _phi(x // PRIMES[a - 1], a - 1)


@lru_cache(maxsize=None)
def prime_pi(n: int) -> int:
    """Lehmer prime counting π(n)."""
    if n < SIEVE_MAX:
        return PI_SMALL[n]

    a = prime_pi(_iroot(n, 4))
    b = prime_pi(int(math.isqrt(n)))
    c = prime_pi(_iroot(n, 3))

    # Lehmer formula
    res = _phi(n, a) + ((b + a - 2) * (b - a + 1)) // 2

    for i in range(a, b):
        p = PRIMES[i]
        w = n // p
        res -= prime_pi(w)
        if i < c:
            lim = prime_pi(int(math.isqrt(w)))
            for j in range(i, lim):
                res -= prime_pi(w // PRIMES[j]) - j
    return res


def _weights_for_N(N: int):
    """w[e] = C(N+e-1, e) mod MOD for e=0..floor(log2 N)."""
    max_e = N.bit_length() - 1
    w = [1] * (max_e + 1)
    cur = 1
    for e in range(1, max_e + 1):
        cur = (cur * ((N + e - 1) % MOD)) % MOD
        cur = (cur * pow(e, MOD - 2, MOD)) % MOD
        w[e] = cur
    return w


def solve(N: int) -> int:
    """
    Compute F(N, N) mod MOD.

    Summatory multiplicative function recursion:
    S(limit, idx) sums f(n) over 1<=n<=limit, where prime factors are >= PRIMES[idx].
    """
    w = _weights_for_N(N)
    w1 = w[1]
    max_e = len(w) - 1

    @lru_cache(maxsize=None)
    def S(limit: int, idx: int) -> int:
        if limit < 2:
            return 1  # only n=1
        if idx >= len(PRIMES) or PRIMES[idx] > limit:
            return 1

        p0 = PRIMES[idx]
        if p0 * p0 > limit:
            # Any number using primes >= p0 is either 1 or a single prime (exponent 1).
            cnt = prime_pi(limit) - prime_pi(p0 - 1)
            return (1 + (cnt % MOD) * w1) % MOD

        res = 1
        root = int(math.isqrt(limit))

        # Numbers that are just one prime > sqrt(limit)
        cnt_large = prime_pi(limit) - prime_pi(root)
        res = (res + (cnt_large % MOD) * w1) % MOD

        i = idx
        while i < len(PRIMES):
            p = PRIMES[i]
            if p > root:
                break
            pe = p
            e = 1
            while pe <= limit:
                res = (res + w[e] * S(limit // pe, i + 1)) % MOD
                e += 1
                if e > max_e:
                    break
                pe *= p
            i += 1

        return res

    return S(N, 0)


def main() -> None:
    # Test values from the problem statement
    assert solve(10) == 571
    assert solve(10**6) == 252903833

    # Required output
    print(solve(10**9))


if __name__ == "__main__":
    main()
