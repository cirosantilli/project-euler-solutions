#!/usr/bin/env python3
"""
Project Euler 802: Iterated Composition

We study the map on R^2:
  f(x, y) = (x^2 - x - y^2, 2xy - y + pi)

Using z = x + i y, this is the complex quadratic polynomial:
  F(z) = z^2 - z + i*pi

The key fact is that the requested sum is always an integer and can be computed
purely arithmetically (pi never needs to be evaluated).
"""

from array import array


MOD = 1_020_340_567
N_DEFAULT = 10_000_000


def _build_floor_div_queries(n: int) -> list[int]:
    """Return sorted distinct values of floor(n / k) for k=1..n."""
    qs = set()
    k = 1
    while k <= n:
        q = n // k
        qs.add(q)
        k = n // q + 1
    return sorted(qs)


def _mertens_at_points(n: int, points: list[int]) -> dict[int, int]:
    """
    Compute Mertens M(t) = sum_{m<=t} mu(m) for selected t in 'points',
    by sieving mu up to n (linear sieve).
    """
    if not points:
        return {}
    points = sorted(set(points))
    res: dict[int, int] = {}

    # mu in {-1,0,1} fits in signed byte; lp is smallest prime factor.
    mu = array("b", [0]) * (n + 1)
    lp = array("I", [0]) * (n + 1)
    primes: list[int] = []

    mu[1] = 1
    mertens = 1

    idx = 0
    if points[idx] == 1:
        res[1] = 1
        idx += 1

    # Linear sieve.
    for i in range(2, n + 1):
        if lp[i] == 0:
            lp[i] = i
            primes.append(i)
            mu[i] = -1

        li = lp[i]
        mui = mu[i]

        for p in primes:
            if p > li:
                break
            ip = i * p
            if ip > n:
                break
            lp[ip] = p
            if p == li:
                mu[ip] = 0
                break
            else:
                mu[ip] = -mui

        mertens += mu[i]
        if idx < len(points) and i == points[idx]:
            res[i] = mertens
            idx += 1

    return res


def _pow2_cached_factory(mod: int):
    cache = {0: 1}

    def pow2(e: int) -> int:
        v = cache.get(e)
        if v is None:
            v = pow(2, e, mod)
            cache[e] = v
        return v

    return pow2


def P_mod(n: int, mod: int = MOD) -> int:
    """
    Return P(n) modulo 'mod', where P(n) is the sum of x-coordinates of all
    points with period <= n.

    Technique:
      - Let A(d) be the sum of x of points with period dividing d.
        For this problem:
            A(1) = 2, and A(d) = 2^(d-1) for d >= 2.
      - Let S(d) be the sum for exact period d. Then A(n) = sum_{d|n} S(d).
        Möbius inversion gives S and yields:
            P(n) = sum_{d<=n} A(d) * M(floor(n/d))
        where M is the Mertens function.
      - Evaluate the sum using floor-division grouping (a.k.a. harmonic lemma).
    """
    if n <= 0:
        return 0

    # We only need M(t) for t in { floor(n/k) }.
    qs = _build_floor_div_queries(n)
    mertens = _mertens_at_points(n, qs)

    pow2 = _pow2_cached_factory(mod)

    def sum_A(l: int, r: int) -> int:
        """
        Sum_{d=l..r} A(d) mod mod, where:
          A(1)=2, A(d)=2^(d-1) for d>=2.
        Convenient closed forms:
          - sum_{d=1..r} A(d) = 2^r
          - for l>=2: sum_{d=l..r} A(d) = 2^r - 2^(l-1)
        """
        if l == 1:
            return pow2(r)
        return (pow2(r) - pow2(l - 1)) % mod

    ans = 0
    l = 1
    while l <= n:
        q = n // l
        r = n // q
        m = mertens[q]  # integer, can be negative
        ans = (ans + sum_A(l, r) * m) % mod
        l = r + 1

    return ans


def main() -> None:
    # Tests from the problem statement
    assert P_mod(1) == 2
    assert P_mod(2) == 2
    assert P_mod(3) == 4

    print(P_mod(N_DEFAULT))


if __name__ == "__main__":
    main()
