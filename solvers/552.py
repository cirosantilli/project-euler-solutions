#!/usr/bin/env python
"""
Project Euler 552: Chinese Leftovers II

Let A_n be the smallest positive integer such that:
    A_n mod p_i = i   for 1 <= i <= n,
where p_i is the i-th prime.

Let S(n) be the sum of all primes <= n that divide at least one A_k.

Compute S(300000).

This program uses only the Python standard library.
"""

from __future__ import annotations

import sys
from typing import List


def primes_up_to(n: int) -> List[int]:
    """Return list of all primes <= n (simple sieve)."""
    if n < 2:
        return []
    sieve = bytearray(b"\x01") * (n + 1)
    sieve[0:2] = b"\x00\x00"
    r = int(n**0.5)
    for i in range(2, r + 1):
        if sieve[i]:
            step = i
            start = i * i
            sieve[start : n + 1 : step] = b"\x00" * (((n - start) // step) + 1)
    return [i for i in range(n + 1) if sieve[i]]


def compute_A(k: int) -> int:
    """
    Compute A_k exactly using incremental CRT:

    Maintain (A, P) where P is product of processed primes.
    When adding constraint x ≡ k (mod p_k):
        A' = A + t*P
        t ≡ (k - A) * inv(P mod p_k) (mod p_k)
    """
    if k <= 0:
        raise ValueError("k must be positive")
    # Need first k primes; k=10 is tiny, so just sieve a safe bound.
    # For k <= 10^5, 2*k*log k would be fine; but here we only use small k in tests.
    primes = primes_up_to(1000)
    if len(primes) < k:
        # Fallback (shouldn't trigger for our test sizes).
        bound = 20000
        while True:
            primes = primes_up_to(bound)
            if len(primes) >= k:
                break
            bound *= 2

    A = 0
    P = 1
    for idx in range(1, k + 1):
        p = primes[idx - 1]
        Ap = A % p
        Pp = P % p
        inv = pow(Pp, p - 2, p)  # p is prime
        t = ((idx - Ap) * inv) % p
        A += t * P
        P *= p
    return A


def compute_S(limit: int, block_size: int = 64) -> int:
    """
    Compute S(limit): sum of primes <= limit dividing at least one A_n.

    Only residues modulo primes <= limit are tracked.  Once a prime has entered
    the CRT system it can no longer divide a later A_n, so at stage i the code
    updates and tests only primes after p_i.

    ``block_size`` is retained for backward-compatible callers; it is unused by
    the residue-array implementation.
    """
    if limit < 2:
        return 0

    primes = primes_up_to(limit)
    m = len(primes)
    if m <= 1:
        return 0

    a_mod = [1 % q for q in primes]  # A_1 mod q
    m_mod = [2 % q for q in primes]  # M_1 mod q
    found = bytearray(m)

    for idx in range(1, m):
        p = primes[idx]
        t = ((idx + 1 - a_mod[idx]) * pow(m_mod[idx], p - 2, p)) % p

        for j in range(idx + 1, m):
            q = primes[j]
            a = (a_mod[j] + m_mod[j] * t) % q
            a_mod[j] = a
            m_mod[j] = (m_mod[j] * p) % q
            if a == 0:
                found[j] = 1

    return sum(q for q, hit in zip(primes, found) if hit)


def _self_test() -> None:
    # Test values explicitly given in the problem statement.
    assert compute_A(2) == 5
    assert compute_A(3) == 23
    assert compute_A(4) == 53
    assert compute_A(5) == 1523
    a10 = compute_A(10)
    assert a10 == 5765999453
    assert a10 % 41 == 0
    assert compute_S(50) == 69


def main() -> None:
    _self_test()
    ans = compute_S(300000)
    print(ans)


if __name__ == "__main__":
    main()
