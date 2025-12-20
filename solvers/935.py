#!/usr/bin/env python3
"""
Project Euler 935: Rolling Square

No external libraries are used (standard library only).

Key idea:
Let the small square have side length 1 and the big square have side length L = 1/b = h + x,
where h is a positive integer and x in [0, 1).

The rolling dynamics induces a 1D map on the leftover length x (see README for details).
All b values that return to the initial position within N rolls fall into:
  1) Integer side lengths L = h (i.e. x = 0), giving 4*(h-1) rolls.
  2) Non-integer L = h + x_{u,v} with gcd(u,v)=1, giving roll counts that reduce to counting
     coprime pairs (u,t) under simple bounds, separated by u mod 4.

We compute F(N) by counting those coprime pairs using Möbius inversion and the
Dirichlet hyperbola method, accelerated by a Du Jiao sieve for the Mertens function.
"""

from __future__ import annotations

from array import array
from typing import Dict, Tuple


def _tri(n: int) -> int:
    """T(n) = 1+2+...+n."""
    return n * (n + 1) // 2


def _build_mu_prefix(limit: int) -> Tuple[array, array, array]:
    """
    Linear sieve for Möbius μ up to 'limit', plus prefix sums:
      M[i]    = sum_{k<=i} μ(k)
      Modd[i] = sum_{k<=i, k odd} μ(k)
    """
    mu = array("b", [0]) * (limit + 1)
    is_comp = bytearray(limit + 1)
    primes = []
    mu[1] = 1
    for i in range(2, limit + 1):
        if not is_comp[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            ip = i * p
            if ip > limit:
                break
            is_comp[ip] = 1
            if i % p == 0:
                mu[ip] = 0
                break
            mu[ip] = -mu[i]

    M = array("i", [0]) * (limit + 1)
    Modd = array("i", [0]) * (limit + 1)

    s = 0
    so = 0
    for i in range(1, limit + 1):
        s += mu[i]
        M[i] = s
        if i & 1:
            so += mu[i]
        Modd[i] = so

    return mu, M, Modd


class RollingSquareCounter:
    """
    Precomputes what is needed to evaluate F(N) up to a chosen max_N.
    """

    def __init__(self, max_N: int):
        self.max_N = max_N
        self.max_X = max_N + 1

        # Du Jiao sieve works well with a pre-sieve up to n^(2/3).
        # A small safety margin helps avoid boundary issues.
        self.limit = int(self.max_X ** (2 / 3)) + 10
        _, self.M_small, self.Modd_small = _build_mu_prefix(self.limit)

        self._cache_M: Dict[int, int] = {}
        self._cache_Modd: Dict[int, int] = {}

    def M(self, n: int) -> int:
        """
        Mertens function M(n) = sum_{k<=n} μ(k), computed by Du Jiao sieve recursion.
        """
        if n <= self.limit:
            return self.M_small[n]
        hit = self._cache_M.get(n)
        if hit is not None:
            return hit

        res = 1
        l = 2
        while l <= n:
            q = n // l
            r = n // q
            res -= (r - l + 1) * self.M(q)
            l = r + 1

        self._cache_M[n] = res
        return res

    def Modd(self, n: int) -> int:
        """
        Odd-restricted Mertens:
          Modd(n) = sum_{k<=n, k odd} μ(k)

        Using μ(4m)=0, we have:
          M(n) = Modd(n) - Modd(floor(n/2))
        hence:
          Modd(n) = M(n) + Modd(floor(n/2))
        """
        if n <= 0:
            return 0
        if n <= self.limit:
            return self.Modd_small[n]
        hit = self._cache_Modd.get(n)
        if hit is not None:
            return hit
        res = self.M(n) + self.Modd(n // 2)
        self._cache_Modd[n] = res
        return res

    def _sum_mu_odd(self, l: int, r: int) -> int:
        return self.Modd(r) - self.Modd(l - 1)

    def _sum_mu_2mod4(self, l: int, r: int) -> int:
        # sum_{d in [l,r], d ≡ 2 (mod 4)} μ(d)
        # For d = 2m with m odd, μ(2m) = -μ(m), otherwise (4|d) => μ(d)=0.
        return -self.Modd(r // 2) + self.Modd((l - 1) // 2)

    def _class_sum(self, X: int, cls: str) -> int:
        """
        Computes:
          S_cls(X) = sum_{u in cls, u<=X} (C(X,u) - φ(u))

        where C(X,u) = #{t<=X : gcd(t,u)=1}, and cls is one of:
          - 'div4'   : u ≡ 0 (mod 4)
          - '2mod4'  : u ≡ 2 (mod 4)
          - 'odd'    : u odd   (includes u=1; removed later)

        Uses Möbius inversion with parity-aware splitting, grouped by q = floor(X/d).
        """
        A = 0  # sum_{u in cls} C(X,u)
        B = 0  # sum_{u in cls} φ(u)

        l = 1
        while l <= X:
            q = X // l
            r = X // q

            odd_mu = self._sum_mu_odd(l, r)
            mu2 = self._sum_mu_2mod4(l, r)

            if cls == "div4":
                # For u ≡ 0 mod 4, dk must be divisible by 4:
                #   d odd  -> k multiple of 4
                #   d ≡ 2  -> k even
                A += q * ((q // 4) * odd_mu + (q // 2) * mu2)
                B += (4 * _tri(q // 4)) * odd_mu + (2 * _tri(q // 2)) * mu2

            elif cls == "2mod4":
                # For u ≡ 2 mod 4, dk ≡ 2 mod 4:
                #   d odd  -> k ≡ 2 mod 4
                #   d ≡ 2  -> k odd
                c = (q + 2) // 4  # count of k <= q with k ≡ 2 mod 4
                A += q * (c * odd_mu + ((q + 1) // 2) * mu2)

                # sum of k ≡ 2 mod 4 up to q is 2*c^2
                B += (2 * c * c) * odd_mu + (_tri(q) - 2 * _tri(q // 2)) * mu2

            elif cls == "odd":
                # For odd u, dk must be odd -> only odd d contribute, and k must be odd.
                A += q * (((q + 1) // 2) * odd_mu)
                B += (_tri(q) - 2 * _tri(q // 2)) * odd_mu

            else:
                raise ValueError(f"Unknown class: {cls}")

            l = r + 1

        return A - B

    def F(self, N: int) -> int:
        """
        Number of b in (0,1) such that the rolling square returns to its initial position
        within N rolls.
        """
        if N < 0:
            return 0
        if N > self.max_N:
            raise ValueError("N exceeds precomputed maximum")

        # Non-integer L = 1/b cases:
        X1 = N + 1
        X2 = N // 2 + 1
        X3 = N // 4 + 1

        res = 0
        res += self._class_sum(X1, "div4")
        res += self._class_sum(X2, "2mod4")
        res += self._class_sum(X3, "odd")

        # Remove u=1 (odd class) which does not correspond to any valid (u,v) with 1<=v<u.
        res -= X3 - 1

        # Integer L = h cases (x = 0): 4*(h-1) rolls, with h >= 2  <=> 1/h < 1.
        res += N // 4

        return res


def solve() -> None:
    N = 10**8
    counter = RollingSquareCounter(N)

    # Test values from the problem statement:
    assert counter.F(6) == 4
    assert counter.F(100) == 805

    print(counter.F(N))


if __name__ == "__main__":
    solve()
