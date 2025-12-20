#!/usr/bin/env python3
"""
Project Euler 787: Bézout's Game

We count winning positions (a,b) with gcd(a,b)=1, a>0, b>0, a+b<=N.

Key derived rule (proved in README):
- If a+b is even: position is always winning.
- If a+b is odd: position is losing iff min(a,b) is even.

Thus:
H(N) = total_coprime_pairs_with_sum_le_N - losing_pairs

Total coprime ordered pairs with a+b<=N is sum_{s=2..N} phi(s).

Losing ordered pairs correspond to:
(a+b odd) and min(a,b) even  <=>  the even number is the smaller one.
These come in symmetric pairs (a,b) and (b,a), so we count unordered then double.

We compute losing count using Möbius inversion over a constrained region.
"""

import sys
from array import array


def _sieve_mu_phi(limit):
    """
    Linear sieve up to `limit` producing:
      mu[n] in {-1,0,1}
      phi[n]
    and prefix sums:
      pre_mu[n]    = sum_{k<=n} mu[k]
      pre_phi[n]   = sum_{k<=n} phi[k]
      pre_mu_odd[n]= sum_{k<=n, k odd} mu[k]
    """
    mu = array("b", [0]) * (limit + 1)
    phi = array("I", [0]) * (limit + 1)
    is_comp = bytearray(limit + 1)
    primes = []

    mu[1] = 1
    phi[1] = 1

    for i in range(2, limit + 1):
        if not is_comp[i]:
            primes.append(i)
            mu[i] = -1
            phi[i] = i - 1
        for p in primes:
            v = i * p
            if v > limit:
                break
            is_comp[v] = 1
            if i % p == 0:
                mu[v] = 0
                phi[v] = phi[i] * p
                break
            else:
                mu[v] = -mu[i]
                phi[v] = phi[i] * (p - 1)

    pre_mu = array("q", [0]) * (limit + 1)
    pre_phi = array("q", [0]) * (limit + 1)
    pre_mu_odd = array("q", [0]) * (limit + 1)

    for i in range(1, limit + 1):
        pre_mu[i] = pre_mu[i - 1] + mu[i]
        pre_phi[i] = pre_phi[i - 1] + phi[i]
        pre_mu_odd[i] = pre_mu_odd[i - 1] + (mu[i] if (i & 1) else 0)

    return pre_mu, pre_phi, pre_mu_odd


def _du_jiao_phi_mu(N):
    """
    Build Du Jiao sieve helpers for summatory:
      S_phi(n) = sum_{k<=n} phi(k)
      S_mu(n)  = sum_{k<=n} mu(k)
    using precomputation up to L ~ N^(2/3).
    """
    L = int(N ** (2.0 / 3.0)) + 10
    if L > N:
        L = N

    pre_mu, pre_phi, pre_mu_odd = _sieve_mu_phi(L)
    memo_mu = {}
    memo_phi = {}

    def S_mu(n):
        if n <= L:
            return int(pre_mu[n])
        if n in memo_mu:
            return memo_mu[n]
        res = 1
        i = 2
        while i <= n:
            q = n // i
            j = n // q
            res -= (j - i + 1) * S_mu(q)
            i = j + 1
        memo_mu[n] = res
        return res

    def S_phi(n):
        if n <= L:
            return int(pre_phi[n])
        if n in memo_phi:
            return memo_phi[n]
        res = n * (n + 1) // 2
        i = 2
        while i <= n:
            q = n // i
            j = n // q
            res -= (j - i + 1) * S_phi(q)
            i = j + 1
        memo_phi[n] = res
        return res

    memo_mu_odd = {}

    def S_mu_odd(n):
        """
        Sum of mu(k) for odd k<=n.
        Identity:
            M(n) = M_odd(n) - M_odd(floor(n/2))
        so:
            M_odd(n) = M(n) + M_odd(floor(n/2))
        """
        if n <= L:
            return int(pre_mu_odd[n])
        if n in memo_mu_odd:
            return memo_mu_odd[n]
        res = S_mu(n) + S_mu_odd(n // 2)
        memo_mu_odd[n] = res
        return res

    # Force-fill memo tables for M(N) once (useful for later queries)
    S_mu(N)

    return S_phi, S_mu_odd


def _C(M):
    """
    Count of (x,y) with:
      x>=1
      y odd >=1
      2x < y
      2x + y <= M

    Closed form:
      let t = floor((M-1)/4)
      C(M) = sum_{x=1..t} (M - 4x + 1)//2
           = t*((M+1)//2) - t*(t+1)
    """
    t = (M - 1) // 4
    if t <= 0:
        return 0
    return t * ((M + 1) // 2) - t * (t + 1)


def H(N):
    """
    Compute H(N) using derived rule + number theory counting.
    """
    S_phi, S_mu_odd = _du_jiao_phi_mu(N)

    # Total coprime ordered pairs with sum<=N is sum_{s=2..N} phi(s) = S_phi(N) - 1
    total_positions = S_phi(N) - 1

    # Losing unordered pairs correspond to:
    # even = 2x, odd = y, gcd(x,y)=1, y odd, 2x<y, 2x+y<=N
    # By Möbius inversion over odd d:
    #   L_u(N) = sum_{d odd} mu(d) * C(floor(N/d))
    losing_unordered = 0

    d = 1
    while d <= N:
        q = N // d
        nd = N // q

        mu_range_odd = S_mu_odd(nd) - S_mu_odd(d - 1)
        losing_unordered += mu_range_odd * _C(q)

        d = nd + 1

    losing_ordered = 2 * losing_unordered
    return total_positions - losing_ordered


def main():
    # Problem statement test values:
    assert H(4) == 5
    assert H(100) == 2043

    N = 10**9
    print(H(N))


if __name__ == "__main__":
    main()
