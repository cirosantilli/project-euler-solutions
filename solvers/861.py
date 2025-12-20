#!/usr/bin/env python3
"""
Project Euler 861

Counts integers 1 < n <= N such that the product of all bi-unitary divisors of n
equals n^k. We need sum_{k=2..10} Q_k(10^12).

Key facts used:
- For n>1, bi-unitary divisors come in complementary pairs (d, n/d) with d != n/d,
  hence the product of all bi-unitary divisors is n^(b(n)/2), where b(n) is the
  number of bi-unitary divisors of n.
- b(n) is multiplicative and for prime powers:
      b(p^e) = e      if e is even
               e + 1  if e is odd
  (equivalently e + (e mod 2)).
- Therefore P(n) = n^k  <=>  b(n) = 2k.

We enumerate possible factor patterns with b(n) in {4,6,8,...,20} by splitting:
    n = a * s
where:
- a is "powerful": every prime exponent >= 2
- s is squarefree (exponents exactly 1), coprime to a

Then:
    b(n) = b(a) * 2^{omega(s)}
So for each powerful a with b(a)=d, we need omega(s)=m such that d*2^m in target set.

To count squarefree s with exactly m primes under a limit, we need many prime
counting queries pi(x). We precompute pi(x) for all x in the hyperbola set
{ floor(N/i) } U { 1..floor(sqrt(N)) } using an optimized "Lucy/hyperbola" sieve.
No external libraries are used.
"""

import math
import sys


# ----------------------------
# Basic sieve up to sqrt(N)
# ----------------------------
def sieve_with_pi(limit: int):
    """Return (primes, pi) for 0..limit using a bytearray sieve."""
    if limit < 2:
        return [], [0] * (limit + 1)
    is_prime = bytearray(b"\x01") * (limit + 1)
    is_prime[0:2] = b"\x00\x00"
    r = int(limit**0.5)
    for i in range(2, r + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start : limit + 1 : step] = b"\x00" * (
                ((limit - start) // step) + 1
            )

    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    pi = [0] * (limit + 1)
    c = 0
    for i in range(limit + 1):
        if is_prime[i]:
            c += 1
        pi[i] = c
    return primes, pi


# -----------------------------------------
# Prime counting table on hyperbola values
# (optimized Lucy / hyperbola method)
# -----------------------------------------
def build_prime_pi_table(N: int, primes, S: int):
    """
    Build array g for values in:
      V = { floor(N/i) for i=1..S } U { 1..start_small }
    stored as:
      indices 0..S-1: value floor(N/(i+1))
      indices S..m-1: small values v descending, where index(v)=m-v
    """
    # If N is a perfect square, floor(N/S)=S duplicates the small value S,
    # so we store small values only down to S-1.
    if N // S == S:
        start_small = S - 1
    else:
        start_small = S

    m = S + start_small
    g = [0] * m

    # Large part: g[i] = floor(N/(i+1)) - 1
    for i in range(S):
        g[i] = N // (i + 1) - 1

    # Small part: for v in 1..start_small, index is m - v
    for v in range(1, start_small + 1):
        g[m - v] = v - 1

    # Main Lucy updates:
    # g[v] -= g[v//p] - g[p-1]  for v >= p^2, iterating primes p.
    # Here g[p-1] is found in the small-value half using index = m-(p-1).
    Nloc = N
    Sloc = S
    mloc = m
    gloc = g
    ss = start_small

    for p in primes:
        p2 = p * p
        if p2 > Nloc:
            break

        # sp = g[p-1]
        # Since p-1 <= S-1 <= start_small, it's in the small half at index m-(p-1).
        sp = gloc[mloc - (p - 1)]

        if p2 <= Sloc:
            # Update ALL large indices 0..S-1 because min large value is floor(N/S) >= S >= p^2
            j = p
            for i in range(Sloc):
                if j <= Sloc:
                    gloc[i] -= gloc[j - 1] - sp
                else:
                    gloc[i] -= gloc[mloc - (Nloc // j)] - sp
                j += p

            # Update small values v descending where v >= p^2
            # value v is stored at index m-v
            if p2 <= ss:
                for v in range(ss, p2 - 1, -1):
                    gloc[mloc - v] -= gloc[mloc - (v // p)] - sp

        else:
            # Only some large indices have value >= p^2:
            # floor(N/(i+1)) >= p^2  <=>  i+1 <= floor(N/p^2)
            end = Nloc // p2
            if end > Sloc:
                end = Sloc

            j = p
            for i in range(end):
                if j <= Sloc:
                    gloc[i] -= gloc[j - 1] - sp
                else:
                    gloc[i] -= gloc[mloc - (Nloc // j)] - sp
                j += p

    return g, start_small, m


# -----------------------------------------
# Core solver: compute all Q_k for k=2..10
# -----------------------------------------
def compute_Qs(N: int):
    S = math.isqrt(N)
    primes, pi_small = sieve_with_pi(S)

    g, start_small, m = build_prime_pi_table(N, primes, S)

    # Prime counting function for values encountered in this problem:
    # - for x <= S: direct from pi_small
    # - for x > S: x must be in { floor(N/i) } with i<=S, so index = (N//x) - 1
    def prime_pi(x: int) -> int:
        if x <= S:
            return pi_small[x]
        return g[N // x - 1]

    primes_list = primes
    plen = len(primes_list)

    # Count squarefree numbers <= limit with exactly mleft primes,
    # all primes strictly increasing starting from primes_list[start_idx],
    # and excluding any primes in forb (tuple).
    #
    # We keep forb very small (primes dividing the powerful part), so `p in forb`
    # is acceptable.
    sys.setrecursionlimit(1000000)

    def count_sqf(limit: int, mleft: int, start_idx: int, forb: tuple) -> int:
        if mleft == 0:
            return 1
        if start_idx >= plen:
            return 0

        if mleft == 1:
            # Count primes q >= primes_list[start_idx], q <= limit, q not in forb
            if limit < primes_list[start_idx]:
                return 0
            base = prime_pi(limit)
            if start_idx > 0:
                base -= pi_small[primes_list[start_idx - 1]]
            # subtract forbidden primes in range
            start_p = primes_list[start_idx]
            for q in forb:
                if start_p <= q <= limit:
                    base -= 1
            return base

        if mleft == 2:
            # Sum over p: count primes q>p with p*q <= limit
            cnt = 0
            for i in range(start_idx, plen):
                p = primes_list[i]
                if p * p > limit:
                    break
                if p in forb:
                    continue
                lim = limit // p
                total = prime_pi(lim) - pi_small[p]  # q > p
                for q in forb:
                    if q > p and q <= lim:
                        total -= 1
                cnt += total
            return cnt

        if mleft == 3:
            cnt = 0
            for i in range(start_idx, plen):
                p = primes_list[i]
                if p * p * p > limit:
                    break
                if p in forb:
                    continue
                cnt += count_sqf(limit // p, 2, i + 1, forb)
            return cnt

        if mleft == 4:
            cnt = 0
            for i in range(start_idx, plen):
                p = primes_list[i]
                if p * p * p * p > limit:
                    break
                if p in forb:
                    continue
                cnt += count_sqf(limit // p, 3, i + 1, forb)
            return cnt

        raise ValueError("mleft too large for this task")

    # Q[k] for k=0..10 (we use 2..10)
    Q = [0] * 11

    # For a powerful number a with bi-unitary divisor count d=b(a),
    # choose m squarefree primes => b(n)=d*2^m => k=b(n)/2.
    def process_powerful(a: int, d: int, primes_used: tuple):
        limit = N // a
        # m up to 4 is enough because d>=1 and we only need d*2^m <= 20
        for mleft in range(5):
            tau = d << mleft  # d * 2^mleft
            if tau > 20:
                break
            if tau >= 4:
                k = tau // 2
                if 2 <= k <= 10:
                    Q[k] += count_sqf(limit, mleft, 0, primes_used)

    # Enumerate all powerful numbers a <= N with b(a) <= 20
    # Exponents >=2, and b(p^e)= e if e even else e+1
    def dfs_powerful(start_idx: int, a: int, d: int, primes_used: tuple):
        process_powerful(a, d, primes_used)

        for i in range(start_idx, plen):
            p = primes_list[i]
            # minimal exponent is 2
            if a * p * p > N:
                break

            # iterate exponents e = 2..20 while p^e <= N/a
            limit = N // a
            p_pow = p * p
            e = 2
            while e <= 20 and p_pow <= limit:
                # b(p^e) = e if even else e+1
                f = e if (e & 1) == 0 else (e + 1)
                new_d = d * f
                if new_d <= 20:
                    dfs_powerful(i + 1, a * p_pow, new_d, primes_used + (p,))
                e += 1
                if e > 20:
                    break
                p_pow *= p

    dfs_powerful(0, 1, 1, ())

    return Q


def solve(N: int) -> int:
    Q = compute_Qs(N)
    return sum(Q[2:11])


# ----------------------------
# Self-checks from the prompt
# ----------------------------
def _tests():
    # Given examples:
    Q100 = compute_Qs(10**2)
    assert Q100[2] == 51

    Q1e6 = compute_Qs(10**6)
    assert Q1e6[6] == 6189


def main():
    _tests()
    N = 10**12
    ans = solve(N)
    print(ans)


if __name__ == "__main__":
    main()
