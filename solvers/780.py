#!/usr/bin/env python3
"""Project Euler 780 — fast computation of G(N) modulo 1_000_000_007.

This program is self-contained (pure Python) and single-core.

It includes:
  - G_fast(N): the intended fast solver (works for N=1_000_000_000)
  - G_reference(N): a slower reference solver (OK up to ~1e5–1e6)

Implementation outline

1) Strip part
   The strip tilings can be rewritten into a symmetric hyperbola sum over
   positive integers (u,v) with u*v <= floor(N/(2*sqrt(3))) (call this limit L):

     strip_sum(N) = 2*D(N/2)
                    + 4 * \sum_{u v <= L} ( \lfloor N/(2*gcd(u,v))\rfloor
                                           - \lfloor sqrt(3) * (u v / gcd(u,v))\rfloor )

   We evaluate the hyperbola sum in ~O(sqrt(L)) by:
     - splitting by v <= sqrt(L) and using symmetry,
     - grouping by d | v, where d = gcd(u,v),
     - filtering gcd(t, v/d) = 1 via inclusion-exclusion (Möbius) over
       squarefree divisors.

   The \lfloor sqrt(3) * (...) \rfloor weighted sums reduce to Beatty-type sums
   \sum_{k<=n} \lfloor k * alpha \rfloor with alpha in Q(sqrt(3)). These are computed
   exactly using the complementary Beatty recursion.

2) Hex correction
   Hex tilings are overcounted and must be corrected.

   Using the multiplicative formula from the prompt and an equivalent geometric
   decomposition, the correction reduces to:

     H(X) = \sum_{m<=X} h(m)
          = D(X) + 2 * \sum_{u>v>=1, Q(u,v)<=X, primitive} D( X // Q(u,v) )

   where Q(u,v)=u^2+uv+v^2, “primitive” means gcd(u,v)=1 and (2u+v) % 3 != 0.

   We evaluate this without iterating all (u,v) by jumping in u over ranges
   where q = X // Q(u,v) is constant, and counting primitive u in each range
   using Möbius inclusion-exclusion plus an extra mod-3 filter.

The fast path comfortably handles N=1_000_000_000 on a single CPU core.

"""

from __future__ import annotations

import sys
from math import gcd, isqrt

MOD = 1_000_000_007


# ------------------------- exact sqrt(3) helpers -------------------------


def floor_sqrt3_mul(n: int) -> int:
    """Return floor(sqrt(3) * n) exactly for integer n >= 0."""
    return isqrt(3 * n * n)


def floor_div_sqrt3(n: int, m: int) -> int:
    """Return floor(n / (m*sqrt(3))) exactly for integers n>=0,m>=1."""
    nn = n * n
    den = 3 * m * m
    # Start from a safe underestimate, then adjust.
    t = isqrt(nn // den)
    while den * (t + 1) * (t + 1) <= nn:
        t += 1
    while den * t * t > nn:
        t -= 1
    return t


# ------------------------- divisor summatory D(n) ------------------------


def divisor_summatory(n: int) -> int:
    """D(n) = sum_{i=1..n} floor(n/i) = sum_{k<=n} tau(k)."""
    res = 0
    i = 1
    while i <= n:
        q = n // i
        j = n // q
        res += q * (j - i + 1)
        i = j + 1
    return res


# -------------------------- sieve: spf + mu ------------------------------


def sieve_mu_spf(n: int):
    """Linear-ish sieve for smallest prime factor and Möbius up to n."""
    spf = [0] * (n + 1)
    mu = [0] * (n + 1)
    primes: list[int] = []
    mu[1] = 1

    for i in range(2, n + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
            mu[i] = -1
        for p in primes:
            v = i * p
            if v > n:
                break
            spf[v] = p
            if i % p == 0:
                mu[v] = 0
                break
            mu[v] = -mu[i]

    # Fill any unset spf entries (only 1 remains).
    if n >= 1 and spf[1] == 0:
        spf[1] = 1

    return primes, spf, mu


def _factor_distinct(x: int, spf: list[int]) -> list[int]:
    """Distinct prime factors of x (x>=1) using spf."""
    ps: list[int] = []
    while x > 1:
        p = spf[x]
        ps.append(p)
        while x % p == 0:
            x //= p
    return ps


def build_squarefree_divs(max_n: int, spf: list[int], mu: list[int]):
    """For each n<=max_n, build list of (d, mu(d)) for squarefree d|n."""
    sf: list[list[tuple[int, int]]] = [None] * (max_n + 1)  # type: ignore
    sf[1] = [(1, 1)]
    for n in range(2, max_n + 1):
        ps = _factor_distinct(n, spf)
        divs = [(1, 1)]
        for p in ps:
            # include / exclude this prime
            divs += [(d * p, -s) for (d, s) in divs]
        sf[n] = divs
    return sf


def build_all_divisors(max_n: int, spf: list[int]):
    """For each n<=max_n, build full divisor list (unordered)."""
    divs: list[list[int]] = [None] * (max_n + 1)  # type: ignore
    divs[1] = [1]
    for n in range(2, max_n + 1):
        x = n
        p = spf[x]
        e = 0
        while x % p == 0:
            x //= p
            e += 1
        base = divs[x]
        out: list[int] = []
        pe = 1
        for _ in range(e + 1):
            for d in base:
                out.append(d * pe)
            pe *= p
        divs[n] = out
    return divs


# ---------------------- Beatty sum in Q(sqrt(3)) -------------------------
# We represent alpha as (a + b*sqrt(3)) / c with integers a (can be negative),
# b>=0, c>0. All operations are exact.


def _alpha_norm(a: int, b: int, c: int) -> tuple[int, int, int]:
    if c < 0:
        a, b, c = -a, -b, -c
    g = gcd(gcd(abs(a), b), c)
    if g > 1:
        a //= g
        b //= g
        c //= g
    return a, b, c


def _floor_qsqrt3(a: int, b: int, c: int) -> int:
    """floor((a + b*sqrt(3))/c) with b>=0,c>0."""
    if b == 0:
        return a // c

    # Underestimate using floor(b*sqrt(3)).
    fb = floor_sqrt3_mul(b)
    x = (a + fb) // c

    bb3 = 3 * b * b

    # Adjust upward if needed.
    while True:
        y = (x + 1) * c - a
        if y <= 0 or y * y <= bb3:
            x += 1
        else:
            break

    # Adjust downward if needed.
    while True:
        y = x * c - a
        if y <= 0 or y * y <= bb3:
            break
        x -= 1

    return x


def _alpha_floor(alpha: tuple[int, int, int]) -> int:
    a, b, c = alpha
    return _floor_qsqrt3(a, b, c)


def _alpha_mul_floor(alpha: tuple[int, int, int], n: int) -> int:
    a, b, c = alpha
    return _floor_qsqrt3(a * n, b * n, c)


def _alpha_sub_int(alpha: tuple[int, int, int], k: int) -> tuple[int, int, int]:
    a, b, c = alpha
    return _alpha_norm(a - k * c, b, c)


def _alpha_div_alpha_minus1(alpha: tuple[int, int, int]) -> tuple[int, int, int]:
    """Return beta = alpha/(alpha-1), assuming 1<alpha<2."""
    a, b, c = alpha
    ac = a - c

    # (a+b√3)/(ac+b√3)
    A = a * ac - 3 * b * b
    B = -b * c
    D = ac * ac - 3 * b * b

    if D < 0:
        A, B, D = -A, -B, -D
    if B < 0:
        A, B = -A, -B

    return _alpha_norm(A, B, D)


def _tri(n: int) -> int:
    return n * (n + 1) // 2


def beatty_sum(alpha: tuple[int, int, int], n: int) -> int:
    """S(n,alpha) = sum_{k=1..n} floor(k*alpha), for alpha>1 irrational."""
    res = 0
    sign = 1

    while n > 0:
        f = _alpha_floor(alpha)
        if f > 1:
            res += sign * (f - 1) * _tri(n)
            alpha = _alpha_sub_int(alpha, f - 1)

        m = _alpha_mul_floor(alpha, n)
        res += sign * _tri(m)

        n = m - n
        if n <= 0:
            break

        alpha = _alpha_div_alpha_minus1(alpha)
        sign = -sign

    return res


def beatty_sqrt3(c: int, n: int) -> int:
    """sum_{k=1..n} floor(k * c*sqrt(3))."""
    if n <= 0:
        return 0
    return beatty_sum((0, c, 1), n)


# --------------------------- fast strip sum ------------------------------


def strip_hyperbola_sum(
    N: int,
    V: int,
    L: int,
    sf_divs: list[list[tuple[int, int]]],
    all_divs: list[list[int]],
) -> tuple[int, int]:
    """Return (S1, S2) where

      S1 = sum_{u v <= L} floor(N/(2*gcd(u,v)))
      S2 = sum_{u v <= L} floor(sqrt(3) * (u v / gcd(u,v)))

    Evaluated in ~O(sqrt(L)).
    """
    s1 = 0
    s2 = 0

    for v in range(1, V + 1):
        Umax = L // v
        dv = all_divs[v]

        for d in dv:
            m = v // d
            hi = Umax // d  # = L // (v*d)
            if hi < m:
                continue

            w = N // (2 * d)
            lo_minus = m - 1

            cnt = 0
            sm = 0
            # Möbius inclusion-exclusion over squarefree divisors of m
            for q, muq in sf_divs[m]:
                cnt += muq * (hi // q - lo_minus // q)

                hiq = hi // q
                loq = lo_minus // q
                if hiq:
                    sm += muq * (beatty_sqrt3(v * q, hiq) - beatty_sqrt3(v * q, loq))

            s1 += w * cnt
            s2 += sm

    # Apply symmetry: sum_{u v<=L} F(u,v)
    # = 2*sum_{v<=V} sum_{u>=v, uv<=L} F(u,v) - sum_{v<=V} F(v,v)
    diag1 = 0
    diag2 = 0
    for v in range(1, V + 1):
        diag1 += N // (2 * v)
        diag2 += floor_sqrt3_mul(v)

    S1 = 2 * s1 - diag1
    S2 = 2 * s2 - diag2
    return S1, S2


# ----------------------- fast hex correction sum -------------------------


def _count_mod3_res(lo: int, hi: int, r: int) -> int:
    """Count t in [lo,hi] with t % 3 == r."""
    if hi < lo:
        return 0
    rem = lo % 3
    delta = (r - rem) % 3
    first = lo + delta
    if first > hi:
        return 0
    return (hi - first) // 3 + 1


def hex_Hsum(
    X: int,
    sf_divs: list[list[tuple[int, int]]],
    D_cache: dict[int, int],
) -> int:
    """Compute H(X) = sum_{m<=X} h(m) using the geometric decomposition:

       H(X) = D(X) + 2 * sum_{u>v>=1, Q(u,v)<=X, primitive} D(X//Q(u,v))

    where Q(u,v)=u^2+uv+v^2, primitive: gcd(u,v)=1 and (2u+v)%3!=0.

    Runs by jumping in u over ranges where q = X//Q(u,v) is constant.
    """
    if X <= 0:
        return 0

    def D(n: int) -> int:
        if n <= 0:
            return 0
        v = D_cache.get(n)
        if v is not None:
            return v
        v = divisor_summatory(n)
        D_cache[n] = v
        return v

    V = isqrt(X)
    extra = 0

    for v in range(1, V + 1):
        disc0 = 4 * X - 3 * v * v
        if disc0 <= 0:
            break
        Umax = (-v + isqrt(disc0)) // 2

        u = v + 1
        sfv = sf_divs[v]
        vmod = v % 3

        while u <= Umax:
            # Q(u,v)
            t = u * u + u * v + v * v
            q = X // t
            if q == 0:
                break

            # Find max u' with same q, via Q(u',v) <= X//q
            T = X // q
            disc = 4 * T - 3 * v * v
            uhi = (-v + isqrt(disc)) // 2
            if uhi > Umax:
                uhi = Umax

            lo1 = u - 1

            # Count gcd(u,v)=1 in [u,uhi]
            total = 0
            for d, mud in sfv:
                total += mud * (uhi // d - lo1 // d)

            # Exclude those with (2u+v)%3==0.
            # For mod 3, 2u+v == 0  <=>  u == v (mod 3).
            if vmod != 0:
                bad = 0
                r = vmod
                for d, mud in sfv:
                    # d|v, so d%3 is 1 or 2 here.
                    dm3 = d % 3
                    inv = 1 if dm3 == 1 else 2
                    tlo = (u + d - 1) // d
                    thi = uhi // d
                    rr = (r * inv) % 3
                    bad += mud * _count_mod3_res(tlo, thi, rr)
                total -= bad

            if total:
                extra += total * D(q)

            u = uhi + 1

    return D(X) + 2 * extra


# ------------------------------- G_fast ----------------------------------


def G_fast(N: int) -> int:
    """Compute G(N) modulo MOD."""
    if N <= 0:
        return 0

    M = N // 2
    # L = floor(N/(2*sqrt(3))) = floor((N/2)/sqrt(3))
    L = floor_div_sqrt3(M, 1)
    V_strip = isqrt(L)

    X = N // 4
    V_hex = isqrt(X) if X > 0 else 0

    max_pre = max(V_strip, V_hex, 1)
    _, spf, mu = sieve_mu_spf(max_pre)
    sf_divs = build_squarefree_divs(max_pre, spf, mu)
    all_divs = build_all_divisors(max_pre, spf)

    base = 2 * divisor_summatory(M)

    S1, S2 = strip_hyperbola_sum(N, V_strip, L, sf_divs, all_divs)
    strip_part = base + 4 * (S1 - S2)

    # Hex correction: subtract 4 * H(N//4)
    D_cache: dict[int, int] = {}
    H = hex_Hsum(X, sf_divs, D_cache)

    total = strip_part - 4 * H
    return total % MOD


# --------------------------- reference solution --------------------------


def sieve_spf(n: int) -> list[int]:
    spf = list(range(n + 1))
    for i in range(2, isqrt(n) + 1):
        if spf[i] == i:
            step = i
            start = i * i
            for j in range(start, n + 1, step):
                if spf[j] == j:
                    spf[j] = i
    return spf


def omega_from_spf(x: int, spf: list[int]) -> int:
    cnt = 0
    while x > 1:
        p = spf[x]
        cnt += 1
        while x % p == 0:
            x //= p
    return cnt


def compute_f_table(limit: int) -> list[int]:
    spf = sieve_spf(limit)
    f = [0] * (limit + 1)
    f[1] = 1
    for x in range(2, limit + 1):
        f[x] = 1 << omega_from_spf(x, spf)
    return f


def hex_factor_value_via_spf(m: int, spf: list[int]) -> int:
    """g(m)=prod_{p^e||m} (e+1)^2 if p%3==1 else (e+1)."""
    val = 1
    x = m
    while x > 1:
        p = spf[x]
        e = 0
        while x % p == 0:
            x //= p
            e += 1
        t = e + 1
        if p % 3 == 1:
            val *= t * t
        else:
            val *= t
    return val


def G_reference(N: int) -> int:
    """Slow reference (roughly O(N)) — use only for small N."""
    if N <= 0:
        return 0

    M = N // 2
    Xmax = floor_div_sqrt3(M, 1)
    Kmax = isqrt(Xmax)

    f = compute_f_table(Xmax)

    total = 2 * divisor_summatory(M)

    for k in range(1, Kmax + 1):
        tmax = M // k
        xmax = floor_div_sqrt3(M, k * k)
        for x in range(1, xmax + 1):
            cutoff = floor_sqrt3_mul(k * x)
            cnt = tmax - cutoff
            if cnt <= 0:
                break
            total += 4 * cnt * f[x]

    Xh = N // 4
    if Xh > 0:
        spf = sieve_spf(Xh)
        hex_sum = 0
        for m in range(1, Xh + 1):
            gval = hex_factor_value_via_spf(m, spf)
            hex_sum += 2 * gval
        total -= 2 * hex_sum

    return total


# ------------------------------- CLI / tests -----------------------------


def _self_test() -> None:
    # Tests mentioned in the prompt (strip+hex combined):
    assert G_fast(6) == 14
    assert G_fast(100) == 8090
    assert G_fast(100_000) % MOD == 645124048


def main(argv: list[str]) -> None:
    _self_test()

    if len(argv) >= 2:
        N = int(argv[1])
    else:
        N = 1_000_000_000

    print(G_fast(N) % MOD)


if __name__ == "__main__":
    main(sys.argv)
