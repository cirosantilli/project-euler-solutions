#!/usr/bin/env python3
"""
Project Euler 715 — Sextuplet Norms

Let f(n) be the number of 6-tuples (x1..x6) with 0 <= xi < n such that:
    gcd(x1^2 + x2^2 + x3^2 + x4^2 + x5^2 + x6^2, n^2) = 1

Define:
    G(n) = sum_{k=1..n} f(k) / (k^2 * phi(k))

We must compute:
    G(10^12) mod 1_000_000_007

Key number-theory reduction:
Let g(n) = f(n) / (n^2 * phi(n)). Then g is multiplicative and:

- For p = 2:      g(2^e) = 2^(3e)
- For odd prime p:
      g(p^e) = p^(3e) - chi(p) * p^(3e-3)
  where chi is the nontrivial Dirichlet character mod 4:
      chi(even)=0, chi(1 mod 4)=+1, chi(3 mod 4)=-1

We compute the summatory function sum_{n<=N} g(n) (mod MOD) using a
Min_25-style recursion and a Lucy–Hedgehog prime-sum sieve for:
    sum_{p<=x} (p^3 - chi(p))
over all x in the standard set of distinct floor divisions.

Constraints:
- No external libraries
- Single-threaded
"""

from math import isqrt

MOD = 1_000_000_007


def chi(n: int) -> int:
    """Dirichlet character modulo 4."""
    if (n & 1) == 0:
        return 0
    return 1 if (n & 3) == 1 else -1


def sum_cubes_1_to(n: int) -> int:
    """Sum_{k=1..n} k^3 mod MOD."""
    a = n * (n + 1) // 2
    return (a * a) % MOD


def prefix_chi_1_to(n: int) -> int:
    """
    Sum_{k=1..n} chi(k).
    Values repeat with period 4, and the partial sums are:
      n mod 4 in {1,2} -> 1
      else -> 0
    """
    if n <= 0:
        return 0
    return 1 if (n & 3) in (1, 2) else 0


def sieve_primes(limit: int):
    """Odd-only sieve up to 'limit' (inclusive)."""
    if limit < 2:
        return []
    size = (limit // 2) + 1
    bs = bytearray(b"\x01") * size
    bs[0] = 0  # 1 is not prime
    r = isqrt(limit)
    for i in range(1, r // 2 + 1):
        if bs[i]:
            p = 2 * i + 1
            start = (p * p) // 2
            step = p
            bs[start::step] = b"\x00" * (((size - 1 - start) // step) + 1)
    primes = [2]
    primes.extend(2 * i + 1 for i in range(1, size) if bs[i])
    return primes


def build_values_and_index(N: int):
    """
    Build the standard list of distinct values of floor(N / i), plus 1..sqrt(N),
    merged into one descending unique list.

    Also build index maps:
      - idx_small[x] for x <= sqrt(N)
      - idx_large[N//x] for x > sqrt(N)
    """
    root = isqrt(N)

    # Distinct floor divisions, descending.
    large = []
    i = 1
    while i <= N:
        v = N // i
        large.append(v)
        i = N // v + 1

    small = list(range(root, 0, -1))

    # Merge descending unique
    values = []
    ia = ib = 0
    la = len(large)
    lb = len(small)
    append = values.append
    while ia < la and ib < lb:
        a = large[ia]
        b = small[ib]
        if a > b:
            append(a)
            ia += 1
        elif a < b:
            append(b)
            ib += 1
        else:
            append(a)
            ia += 1
            ib += 1
    if ia < la:
        values.extend(large[ia:])
    if ib < lb:
        values.extend(small[ib:])

    idx_small = [0] * (root + 1)
    idx_large = [0] * (root + 1)
    for idx, v in enumerate(values):
        if v <= root:
            idx_small[v] = idx
        else:
            idx_large[N // v] = idx

    return values, root, idx_small, idx_large


def compute_prime_sums_gprime(N: int, primes):
    """
    Lucy–Hedgehog prime-sum sieve over all needed x in the 'values' list.

    Produces:
        Gp(x) = sum_{p<=x} (p^3 - chi(p)) mod MOD
    for each x in values.

    Returns:
        root, idx_small, idx_large, Gp_list, primes
    """
    values, root, idx_small, idx_large = build_values_and_index(N)
    m = len(values)

    # Initialize as sums over integers 2..v, then sieve down to primes.
    gcube = [0] * m  # sum i^3
    gchi = [0] * m  # sum chi(i)
    for i, v in enumerate(values):
        gcube[i] = (sum_cubes_1_to(v) - 1) % MOD  # 2..v
        gchi[i] = (prefix_chi_1_to(v) - 1) % MOD  # 2..v

    limit = m  # we only need to update indices where values[idx] >= p^2

    for p in primes:
        p2 = p * p
        if p2 > N:
            break

        while limit > 0 and values[limit - 1] < p2:
            limit -= 1

        ipm1 = idx_small[p - 1]  # p-1 <= root
        gc_pm1 = gcube[ipm1]
        gh_pm1 = gchi[ipm1]

        p3 = (p * p % MOD) * p % MOD
        chip = chi(p)

        for i in range(limit):
            v = values[i]
            u = v // p
            iu = idx_small[u] if u <= root else idx_large[N // u]

            gcube[i] = (gcube[i] - p3 * ((gcube[iu] - gc_pm1) % MOD)) % MOD
            if chip:
                gchi[i] = (gchi[i] - chip * ((gchi[iu] - gh_pm1) % MOD)) % MOD

    gprime = [(gcube[i] - gchi[i]) % MOD for i in range(m)]
    return root, idx_small, idx_large, gprime, primes


def solve(N: int) -> int:
    """
    Compute G(N) = sum_{n<=N} g(n) mod MOD.
    """
    root = isqrt(N)
    primes = sieve_primes(root)
    root, idx_small, idx_large, gprime, primes = compute_prime_sums_gprime(N, primes)

    P = len(primes)
    base = P + 1  # for compact memo keys

    def idx_of(x: int) -> int:
        if x <= root:
            return idx_small[x]
        return idx_large[N // x]

    def prime_sum_upto(x: int) -> int:
        if x < 2:
            return 0
        return gprime[idx_of(x)]

    def prime_sum_range(lo: int, hi: int) -> int:
        """Sum over primes p with lo < p <= hi of (p^3 - chi(p)) mod MOD."""
        if hi <= lo:
            return 0
        return (prime_sum_upto(hi) - prime_sum_upto(lo)) % MOD

    memo = {}

    def S(n: int, idx: int) -> int:
        """
        Summatory over numbers m <= n whose prime factors are >= primes[idx]:
            sum g(m)
        Includes m=1.
        """
        if n < 2:
            return 1

        if idx >= P:
            # only primes > last listed prime can appear (besides 1)
            lo = primes[P - 1]
            return (1 + prime_sum_range(lo, n)) % MOD

        p0 = primes[idx]
        if p0 > n:
            return 1

        key = n * base + idx
        if key in memo:
            return memo[key]

        lo = primes[idx - 1] if idx > 0 else 1

        # If p0^2 > n, there are no composites built from primes >= p0.
        if p0 * p0 > n:
            res = (1 + prime_sum_range(lo, n)) % MOD
            memo[key] = res
            return res

        res = (1 + prime_sum_range(lo, n)) % MOD

        for j in range(idx, P):
            p = primes[j]
            pp = p * p
            if pp > n:
                break

            # exponent 1: composite part only (avoid counting the prime itself twice)
            gp1 = ((p * p % MOD) * p - chi(p)) % MOD  # g(p)
            res = (res + gp1 * ((S(n // p, j + 1) - 1) % MOD)) % MOD

            # exponents >= 2
            sign = chi(p)
            p3 = (p * p % MOD) * p % MOD
            prev = p3  # p^(3*1)
            cur = (prev * p3) % MOD  # p^(3*2)
            pe_int = pp

            while pe_int <= n:
                gp = (cur - sign * prev) % MOD  # g(p^e)
                res = (res + gp * S(n // pe_int, j + 1)) % MOD

                if pe_int > n // p:
                    break
                pe_int *= p
                prev, cur = cur, (cur * p3) % MOD

        memo[key] = res
        return res

    return S(N, 0) % MOD


def main():
    # Test values from the problem statement
    assert solve(10) == 3053
    assert solve(10**5) == 157612967

    print(solve(10**12))


if __name__ == "__main__":
    main()
