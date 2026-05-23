#!/usr/bin/env python
"""
Project Euler 850 - Fractions of Powers

We need:
  f_k(n) = sum_{i=1}^{n} { i^k / n }
  S(N)   = sum_{1<=k<=N, k odd} sum_{1<=n<=N} f_k(n)
Compute floor(S(33557799775533)) mod 977676779.

No external libraries are used (only Python standard library).
"""

from __future__ import annotations

import math


MOD = 977676779
MOD2 = 2 * MOD


def two_f_k_n(k: int, n: int) -> int:
    """Return 2*f_k(n) for odd k, as an integer (avoids floats)."""
    assert k > 0 and n > 0
    if k % 2 == 0:
        raise ValueError("two_f_k_n is intended for odd k only in this solution.")

    # Factor n by trial division (sufficient for the tiny statement examples).
    x = n
    m = 1  # m(n,k) = product p^{ceil(e/k)}
    p = 2
    while p * p <= x:
        if x % p == 0:
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            m *= p ** ((e + k - 1) // k)
        p += 1 if p == 2 else 2
    if x > 1:
        m *= x  # exponent is 1 => ceil(1/k)=1

    # 2*f_k(n) = n - n/m
    return n - (n // m)


def primes_upto(limit: int) -> list[int]:
    if limit < 2:
        return []
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[0:2] = b"\x00\x00"
    for p in range(2, math.isqrt(limit) + 1):
        if sieve[p]:
            start = p * p
            sieve[start : limit + 1 : p] = b"\x00" * ((limit - start) // p + 1)
    return [p for p in range(2, limit + 1) if sieve[p]]


def odd_count_from(first: int, limit: int) -> int:
    if first > limit:
        return 0
    if first % 2 == 0:
        first += 1
    if first > limit:
        return 0
    return (limit - first) // 2 + 1


def compute_twoS_mod(N: int, mod2: int) -> int:
    """Compute 2*S(N) modulo mod2."""
    if N <= 0:
        return 0

    primes = primes_upto(math.isqrt(N))
    factors: list[tuple[int, int]] = []
    H_ge3 = 0

    def profile_contribution(d: int, rad: int, phi_mod: int, max_exp: int) -> int:
        tail_first = max(3, max_exp + 1)
        if tail_first % 2 == 0:
            tail_first += 1
        total = (odd_count_from(tail_first, N) % mod2) * (N // (d * rad)) % mod2

        small_last = min(N, tail_first - 2)
        limit = N // d
        for k in range(3, small_last + 1, 2):
            rho = 1
            denom = k - 1
            for p, exponent in factors:
                rho *= p ** ((exponent + denom - 1) // denom)
                if rho > limit:
                    break
            if rho <= limit:
                total += N // (d * rho)
        return phi_mod * (total % mod2) % mod2

    def visit(start_idx: int, d: int, rad: int, phi_mod: int, max_exp: int) -> None:
        nonlocal H_ge3
        H_ge3 = (H_ge3 + profile_contribution(d, rad, phi_mod, max_exp)) % mod2

        for i in range(start_idx, len(primes)):
            p = primes[i]
            if d * rad * p * p > N:
                break

            next_rad = rad * p
            next_d = d * p
            phi_factor = p - 1
            exponent = 1
            while next_d <= N // next_rad:
                factors.append((p, exponent))
                visit(
                    i + 1,
                    next_d,
                    next_rad,
                    (phi_mod * (phi_factor % mod2)) % mod2,
                    max(max_exp, exponent),
                )
                factors.pop()

                exponent += 1
                next_d *= p
                phi_factor *= p

    visit(0, 1, 1, 1, 0)

    # Assemble Total_h = sum_{odd k<=N} H_k modulo mod2. For k=1,
    # m(n,1)=n, so h_1(n)=1 and H_1=N.
    H1 = N % mod2
    odd_count = (N + 1) // 2  # number of odd k in [1..N]
    total_h = (H1 + H_ge3) % mod2

    sum_n = (N * (N + 1) // 2) % mod2
    # 2S = sum_{odd k} (sum_n - H_k)
    twoS = ((odd_count % mod2) * sum_n - total_h) % mod2
    return twoS


def solve(N: int) -> int:
    """Return floor(S(N)) modulo MOD."""
    twoS_mod = compute_twoS_mod(N, MOD2)
    return (twoS_mod // 2) % MOD


def main() -> None:
    # Asserts for test values given in the problem statement (avoid floats).
    assert two_f_k_n(5, 10) == 9
    assert two_f_k_n(7, 1234) == 1233
    assert compute_twoS_mod(10, MOD2) == 201
    assert compute_twoS_mod(10**3, MOD2) == 247375608

    N = 33557799775533
    print(solve(N))


if __name__ == "__main__":
    main()
