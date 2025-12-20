#!/usr/bin/env python3
"""Project Euler 875: Quadruple Congruence

We count solutions to
  a1^2 + a2^2 + a3^2 + a4^2 ≡ b1^2 + b2^2 + b3^2 + b4^2 (mod n)
with 0 <= ai, bi < n.

Let q(n) be the number of solutions, and Q(N)=sum_{i=1..N} q(i).
We need Q(12345678) modulo 1001961001.

No external libraries are used.
"""

from array import array
import sys

MOD = 1001961001
TARGET_N = 12345678


def q_prime_power_mod(p: int, k: int, mod: int = MOD) -> int:
    """Return q(p^k) modulo mod, where p is prime and k>=1.

    Using Gauss-sum magnitude classification, one can show:
      For odd p:
        q(p^k) = p^(7k) + (p-1)*p^(4k-1) * sum_{j=0..k-1} p^(3j)
      For p=2:
        q(2) = 128
        For k>=2:
          q(2^k) = 2^(7k) + 2^(4k+3) * sum_{j=0..k-2} 2^(3j)

    These identities are integer-valued; we compute them directly modulo mod
    without any modular division.
    """

    if p == 2:
        if k == 1:
            return 128 % mod
        # Geometric series: 1 + 2^3 + ... + 2^{3(k-2)}
        r = 8 % mod
        g = 0
        cur = 1
        for _ in range(k - 1):
            g += cur
            if g >= mod:
                g -= mod
            cur = (cur * r) % mod
        term1 = pow(2, 7 * k, mod)
        term2 = (pow(2, 4 * k + 3, mod) * g) % mod
        return (term1 + term2) % mod

    # Odd prime
    if k == 1:
        # q(p) = p^7 + p^4 - p^3
        p2 = (p * p) % mod
        p3 = (p2 * p) % mod
        p4 = (p2 * p2) % mod
        p7 = (p4 * p3) % mod
        return (p7 + p4 - p3) % mod

    r = pow(p, 3, mod)
    g = 0
    cur = 1
    for _ in range(k):
        g += cur
        if g >= mod:
            g -= mod
        cur = (cur * r) % mod

    term1 = pow(p, 7 * k, mod)
    termp = pow(p, 4 * k - 1, mod)
    term2 = ((p - 1) % mod) * termp % mod * g % mod
    return (term1 + term2) % mod


def smallest_prime_factors(n: int) -> array:
    """Linear sieve: returns array spf where spf[x] is the smallest prime factor of x (spf[1]=1)."""
    spf = array("I", [0]) * (n + 1)
    spf[1] = 1
    primes = []

    spf_local = spf
    primes_append = primes.append
    limit = n

    for i in range(2, limit + 1):
        if spf_local[i] == 0:
            spf_local[i] = i
            primes_append(i)
        si = spf_local[i]
        for p in primes:
            ip = i * p
            if ip > limit:
                break
            spf_local[ip] = p
            if p == si:
                break

    return spf


def compute_Q(N: int, mod: int = MOD) -> int:
    """Compute Q(N) = sum_{n=1..N} q(n) modulo mod."""
    spf = smallest_prime_factors(N)

    f = array("I", [0]) * (N + 1)  # f[n] = q(n) mod mod
    f[1] = 1

    spf_local = spf
    f_local = f
    qpp = q_prime_power_mod
    total = 1 % mod

    for n in range(2, N + 1):
        p = spf_local[n]
        m = n // p

        if spf_local[m] != p:
            # p divides n exactly once
            if m == 1:
                qp = qpp(p, 1, mod)
            else:
                qp = f_local[p]  # already computed because p < n
            val = (f_local[m] * qp) % mod
        else:
            # p divides n at least twice
            ppow = p * p
            k = 2
            mm = m // p
            while mm > 1 and spf_local[mm] == p:
                ppow *= p
                k += 1
                mm //= p
            rest = mm
            if rest == 1:
                qp = qpp(p, k, mod)
            else:
                qp = f_local[ppow]  # ppow < n in this branch
            val = (f_local[rest] * qp) % mod

        f_local[n] = val
        total += val
        if total >= mod:
            total -= mod

    return total


def _self_test() -> None:
    # From the problem statement:
    assert q_prime_power_mod(2, 2, MOD) == 18432
    assert compute_Q(10, MOD) == 18573381


def main() -> None:
    _self_test()

    N = TARGET_N
    if len(sys.argv) >= 2:
        N = int(sys.argv[1])

    print(compute_Q(N, MOD) % MOD)


if __name__ == "__main__":
    main()
