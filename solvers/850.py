#!/usr/bin/env python3
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
from array import array
from bisect import bisect_left


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


def _sieve_mu_prefix(limit: int):
    """Linear sieve up to limit producing primes, spf, mu, prefix_mu, prefix_sqfree."""
    spf = array("I", [0]) * (limit + 1)
    mu = array("b", [0]) * (limit + 1)
    primes = []

    mu[1] = 1
    for i in range(2, limit + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
            mu[i] = -1
        for p in primes:
            # Standard linear-sieve guard: only iterate primes up to spf[i]
            if p > spf[i]:
                break
            ip = i * p
            if ip > limit:
                break
            spf[ip] = p
            if i % p == 0:
                mu[ip] = 0
                break
            mu[ip] = -mu[i]

    prefix_mu = array("i", [0]) * (limit + 1)
    prefix_sqfree = array("i", [0]) * (limit + 1)

    s_mu = 0
    s_sf = 0
    for i in range(1, limit + 1):
        s_mu += int(mu[i])
        prefix_mu[i] = s_mu
        if mu[i] != 0:
            s_sf += 1
        prefix_sqfree[i] = s_sf

    return primes, spf, mu, prefix_mu, prefix_sqfree


def compute_twoS_mod(N: int, mod2: int) -> int:
    """Compute 2*S(N) modulo mod2."""
    if N <= 0:
        return 0

    # Maximum prime exponent among n<=N occurs at p=2.
    max_e_global = N.bit_length() - 1  # floor(log2(N))

    # For odd k > max_e_global, ceil(e/k)=1 for every exponent e>=1, hence
    # h_k(n)=n/rad(n) for all n<=N (k-independent tail).
    tail_first = max_e_global + 1
    if tail_first % 2 == 0:
        tail_first += 1  # smallest odd k > max_e_global

    # We'll compute H_k for odd k in:
    #   k=1 separately, plus k in k_list (odd >=3 and < tail_first),
    # and use H_inf for the remaining tail k values.
    k_list = list(range(3, min(N, tail_first - 2) + 1, 2))
    K = len(k_list)

    # Precompute sieve data up to sqrt(N).
    limit = math.isqrt(N)
    primes, spf, mu, prefix_mu, prefix_sqfree = _sieve_mu_prefix(limit)

    sqfree_cache: dict[int, int] = {}

    def squarefree_count(x: int) -> int:
        """Q(x) = #squarefree <= x."""
        if x <= 0:
            return 0
        if x <= limit:
            return int(prefix_sqfree[x])
        v = sqfree_cache.get(x)
        if v is not None:
            return v

        # Q(x) = sum_{d<=sqrt(x)} mu(d) * floor(x/d^2), grouped by equal quotients
        r = math.isqrt(x)
        total = 0
        i = 1
        while i <= r:
            q = x // (i * i)
            j = math.isqrt(x // q)
            total += (int(prefix_mu[j]) - int(prefix_mu[i - 1])) * q
            i = j + 1

        sqfree_cache[x] = total
        return total

    def count_sqfree_coprime(x: int, rad_primes: list[int]) -> int:
        """
        Count squarefree u <= x with gcd(u, R)=1 where R=prod(rad_primes).

        IMPORTANT: This is NOT plain inclusion–exclusion with Q(x//d) over d|R.
        For squarefree counting, “squarefree and divisible by p” is not Q(x//p),
        because Q(x//p) also includes numbers divisible by p (which would become p^2
        after multiplying back).

        Correct filter for one prime p:
            count = sum_{j>=0} (-1)^j * Q(floor(x / p^j))

        For multiple primes, apply the filters multiplicatively, i.e. sum over
            d = prod p_i^{e_i} with e_i>=0,
        sign = (-1)^{sum e_i}, term = Q(floor(x/d)).
        The sum truncates naturally when d > x.
        """
        if x <= 0:
            return 0
        if not rad_primes:
            return squarefree_count(x)

        total = 0

        def dfs(idx: int, prod: int, sign: int) -> None:
            nonlocal total
            if idx == len(rad_primes):
                total += sign * squarefree_count(x // prod)
                return
            p = rad_primes[idx]
            pe = 1
            s = sign
            while prod * pe <= x:
                dfs(idx + 1, prod * pe, s)
                pe *= p
                s = -s

        dfs(0, 1, 1)
        return total

    # Precompute exponent table for h_k on prime powers.
    # For a prime power p^e (e>=2), h_k contributes p^{e - ceil(e/k)}.
    exp_table: list[list[int]] = []
    if K:
        exp_table = [[0] * K for _ in range(max_e_global + 1)]
        for e in range(2, max_e_global + 1):
            row = exp_table[e]
            for idx_k, k in enumerate(k_list):
                row[idx_k] = e - ((e + k - 1) // k)

    # For each exponent e, find first index in k_list where k >= e.
    # For positions >= that index, the contribution for p^e is constant p^{e-1}.
    idx_for_e = [0] * (max_e_global + 2)
    for e in range(max_e_global + 2):
        idx_for_e[e] = bisect_left(k_list, e) if K else 0

    # H_small[pos] accumulates H_k for k = k_list[pos]
    H_small = [0] * K
    # diff for range-adding constant suffix contributions into H_small
    diff = [0] * (K + 1)
    # H_inf is H_k for odd k >= tail_first (k-independent tail)
    H_inf = 0

    # --- Process powerful part P=1 (i.e. squarefree n) ---
    # For k>=3, h_k(n)=1 when n is squarefree.
    C1 = squarefree_count(N)
    C1m = C1 % mod2
    H_inf = (H_inf + C1m) % mod2
    if K:
        diff[0] += C1m
        diff[K] -= C1m

    # --- Enumerate powerful numbers P>1 via DFS over prime powers p^e (e>=2) ---
    fac_primes: list[int] = []
    fac_exps: list[int] = []

    def process_node(P_val: int, const_mod: int, max_e: int):
        nonlocal H_inf
        x = N // P_val
        # Count u <= x squarefree and coprime to rad(P) = product(fac_primes)
        C = count_sqfree_coprime(x, fac_primes)
        Cm = C % mod2

        # For tail k (and for k>=max_e), h_k(P)=P/rad(P) = const_mod
        amt_const = (const_mod * Cm) % mod2
        H_inf += amt_const
        if H_inf >= mod2:
            H_inf -= mod2

        if not K:
            return

        suffix = idx_for_e[max_e] if max_e < len(idx_for_e) else K
        diff[suffix] += amt_const
        diff[K] -= amt_const

        # Only for k < max_e do we need the exact h_k(P)
        for pos in range(suffix):
            hk = 1
            for t in range(len(fac_primes)):
                p = fac_primes[t]
                e = fac_exps[t]
                hk = (hk * pow(p, exp_table[e][pos], mod2)) % mod2
            H_small[pos] = (H_small[pos] + hk * Cm) % mod2

    def dfs(start_idx: int, cur_val: int, const_mod: int, max_e: int):
        for i in range(start_idx, len(primes)):
            p = primes[i]
            p2 = p * p
            if cur_val * p2 > N:
                break

            val = cur_val * p2
            e = 2
            # For P/rad(P), prime p contributes p^(e-1).
            pow_const = p % mod2  # p^(2-1)

            while val <= N:
                fac_primes.append(p)
                fac_exps.append(e)

                new_const = (const_mod * pow_const) % mod2
                new_max_e = max_e if max_e >= e else e

                process_node(val, new_const, new_max_e)
                dfs(i + 1, val, new_const, new_max_e)

                fac_primes.pop()
                fac_exps.pop()

                e += 1
                val *= p
                pow_const = (pow_const * p) % mod2

    dfs(0, 1, 1, 0)

    # Apply suffix range-adds to H_small.
    if K:
        running = 0
        for pos in range(K):
            running = (running + diff[pos]) % mod2
            H_small[pos] = (H_small[pos] + running) % mod2

    # Assemble Total_h = sum_{odd k<=N} H_k mod mod2.
    # For k=1: m(n,1)=n => h_1(n)=1, so H_1 = N.
    H1 = N % mod2

    sum_H_small = sum(H_small) % mod2
    odd_count = (N + 1) // 2  # number of odd k in [1..N]
    small_count = 1 + K  # k=1 plus the k_list entries
    tail_count = odd_count - small_count

    total_h = (H1 + sum_H_small + (tail_count % mod2) * (H_inf % mod2)) % mod2

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
