#!/usr/bin/env python3
"""
Project Euler 784: Reciprocal Pairs

Compute F(N): sum of (p+q) over all reciprocal pairs (p,q) with p <= N.

No external libraries are used.
"""

from __future__ import annotations


def sieve_spf(limit: int) -> list[int]:
    """Smallest prime factor sieve (linear time)."""
    spf = [0] * (limit + 1)
    spf[1] = 1
    primes: list[int] = []
    for i in range(2, limit + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
        # linear sieve core
        for p in primes:
            ip = i * p
            if ip > limit or p > spf[i]:
                break
            spf[ip] = p
    return spf


def F(N: int) -> int:
    """
    Efficiently compute F(N).

    Key derived form:
      For each r >= 2, let n = r^2 - 1.
      For every divisor k of n with 1 <= k <= min(r-1, N-r), set l = n/k and
      (p,q) = (r+k, r+l) is a reciprocal pair with p <= N.
      Contribution is p+q = 2r + k + l.
    """
    if N <= 2:
        return 0

    spf = sieve_spf(N + 1)
    total = 0

    # r must satisfy r < p <= N and p = r + k with k >= 1, so r <= N-1
    for r in range(2, N):
        kmax = N - r
        if kmax <= 0:
            continue
        r1 = r - 1
        if kmax > r1:
            kmax = r1

        n_val = r * r - 1
        base = 2 * r

        # micro-fast path: only divisor is 1
        if kmax == 1:
            total += base + 1 + n_val
            continue

        a = r - 1
        b = r + 1

        factors: list[tuple[int, int]] = []

        # (r-1, r+1) share only powers of 2 when r is odd (gcd = 2)
        if r & 1:
            ea = (a & -a).bit_length() - 1
            eb = (b & -b).bit_length() - 1
            e2 = ea + eb
            a >>= ea
            b >>= eb
            if 2 <= kmax:
                factors.append((2, e2))

        # Factorise a, keeping only primes <= kmax (others can't appear in divisors <= kmax)
        x = a
        while x > 1:
            p = spf[x]
            if p > kmax:
                break
            e = 1
            x //= p
            while x > 1 and spf[x] == p:
                e += 1
                x //= p
            factors.append((p, e))

        # Factorise b similarly
        x = b
        while x > 1:
            p = spf[x]
            if p > kmax:
                break
            e = 1
            x //= p
            while x > 1 and spf[x] == p:
                e += 1
                x //= p
            factors.append((p, e))

        # Generate all divisors <= kmax from the collected prime powers.
        divs = [1]
        for p, e in factors:
            prev = divs
            new: list[int] = []
            new_append = new.append
            pow_p = 1
            for _ in range(e + 1):
                for d in prev:
                    v = d * pow_p
                    if v <= kmax:
                        new_append(v)
                pow_p *= p
                if pow_p > kmax:
                    break
            divs = new

        for k in divs:
            total += base + k + (n_val // k)

    return total


def main() -> None:
    # Tests given in the problem statement:
    assert F(5) == 59
    assert F(10**2) == 697317

    print(F(2 * 10**6))


if __name__ == "__main__":
    main()
