#!/usr/bin/env python3
"""Project Euler 795: Alternating GCD Sum

For a positive integer n:
    g(n) = sum_{i=1..n} (-1)^i * gcd(n, i^2)
    G(N) = sum_{n=1..N} g(n)

This script computes G(12345678).

Constraints:
- No external libraries.
- Asserts include all check values stated in the problem.
- The final Project Euler answer is not hard-coded; it is computed and printed.
"""

from __future__ import annotations

from array import array
import sys


def a_prime_power(p: int, e: int) -> int:
    """A(p^e) for the multiplicative function A used in the solution.

    The closed forms come from summing a divisor expression over p-adic valuations.
    """
    if e == 0:
        return 1

    if e & 1:
        # e = 2k+1
        k = e >> 1
        # p^{2k} * (2*p^{k+1} - 1)
        return pow(p, e - 1) * (2 * pow(p, k + 1) - 1)
    else:
        # e = 2k
        k = e >> 1
        # p^{2k-1} * ((p+1)*p^k - 1)
        return pow(p, e - 1) * ((p + 1) * pow(p, k) - 1)


def smallest_prime_factors(limit: int) -> array:
    """Linear sieve for smallest prime factors up to 'limit' (inclusive)."""
    spf = array("I", [0]) * (limit + 1)
    if limit < 2:
        return spf

    primes: list[int] = []
    primes_append = primes.append
    spf_local = spf

    for i in range(2, limit + 1):
        if spf_local[i] == 0:
            spf_local[i] = i
            primes_append(i)
        # Each composite is set once.
        for p in primes:
            x = p * i
            if x > limit:
                break
            spf_local[x] = p
            if p == spf_local[i]:
                break

    return spf


def a_of_odd(n: int, spf: array) -> int:
    """Compute A(n) for odd n >= 1 using its prime factorization."""
    if n == 1:
        return 1

    res = 1
    while n > 1:
        p = spf[n]
        e = 1
        n //= p
        while n % p == 0:
            n //= p
            e += 1
        res *= a_prime_power(p, e)
    return res


def prepare_c2(N: int) -> list[int]:
    """Precompute c2[a] = A(2^a) - 2^a for all relevant a."""
    max_a = N.bit_length()
    c2 = [0] * (max_a + 1)
    for a in range(1, max_a + 1):
        if (1 << a) > N:
            break
        c2[a] = a_prime_power(2, a) - (1 << a)
    return c2


def g_value(n: int, spf: array, c2: list[int]) -> int:
    """Compute g(n)."""
    if n & 1:
        return -n

    # n = 2^a * m with m odd.
    a = (n & -n).bit_length() - 1
    m = n >> a
    a_m = 1 if m == 1 else a_of_odd(m, spf)
    return a_m * c2[a]


def compute_G(N: int) -> int:
    """Compute G(N) = sum_{n=1..N} g(n)."""
    if N <= 0:
        return 0

    # For any n <= N, the odd part of n is <= N/2 when n is even.
    # We only need factorization up to N//2.
    limit = N // 2
    spf = smallest_prime_factors(limit)
    c2 = prepare_c2(N)

    # ---- asserts from the problem statement ----
    assert g_value(4, spf, c2) == 6
    assert g_value(1234, spf, c2) == 1233
    s = 0
    for k in range(1, 1234 + 1):
        s += g_value(k, spf, c2)
    assert s == 2194708

    # ---- fast summation ----
    # For odd n, g(n) = -n, hence the odd contribution is -sum(odd numbers <= N).
    odd_cnt = (N + 1) // 2
    total = -(odd_cnt * odd_cnt)  # 1+3+...+(2*odd_cnt-1) = odd_cnt^2

    # Enumerate even n uniquely as n = m * 2^a with m odd and a >= 1.
    for m in range(1, limit + 1, 2):
        a_m = 1 if m == 1 else a_of_odd(m, spf)
        a = 1
        n = m << 1
        while n <= N:
            total += a_m * c2[a]
            a += 1
            n <<= 1

    return total


def main() -> None:
    N = 12345678
    if len(sys.argv) > 1:
        N = int(sys.argv[1])
    print(compute_G(N))


if __name__ == "__main__":
    main()
