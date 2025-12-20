#!/usr/bin/env python3
"""Project Euler 764: Asymmetric Diophantine Equation

We need S(N) = sum(x+y+z) over all primitive solutions (x,y,z) with
    16*x^2 + y^4 = z^2,
    1 <= x,y,z <= N,
    gcd(x,y,z) = 1.

The program computes S(10^16) modulo 10^9.

No external libraries are used.
"""

from __future__ import annotations

from math import gcd, isqrt


MOD_DEFAULT = 1_000_000_000


def iroot4(n: int) -> int:
    """floor(n^(1/4)) for n>=0."""
    if n <= 0:
        return 0
    return isqrt(isqrt(n))


def ceil_root4(n: int) -> int:
    """ceil(n^(1/4)) for n>=0."""
    if n <= 0:
        return 0
    r = iroot4(n)
    if r * r * r * r < n:
        r += 1
    return r


def odd_count(n: int) -> int:
    """Count of positive odd integers <= n."""
    if n <= 0:
        return 0
    return (n + 1) // 2


def odd_sum1(n: int) -> int:
    """Sum of positive odd integers <= n."""
    if n <= 0:
        return 0
    m = (n + 1) // 2
    return m * m


def odd_sum4(n: int) -> int:
    """Sum of fourth powers of positive odd integers <= n."""
    if n <= 0:
        return 0
    m = (n + 1) // 2  # odds: 1,3,...,(2m-1)

    # Power sums over i=1..m
    s1 = m * (m + 1) // 2
    s2 = m * (m + 1) * (2 * m + 1) // 6
    s3 = s1 * s1
    s4 = m * (m + 1) * (2 * m + 1) * (3 * m * m + 3 * m - 1) // 30

    # (2i-1)^4 = 16i^4 - 32i^3 + 24i^2 - 8i + 1
    return 16 * s4 - 32 * s3 + 24 * s2 - 8 * s1 + m


def sieve_spf(n: int) -> list[int]:
    """Smallest prime factor sieve up to n."""
    spf = list(range(n + 1))
    limit = int(n**0.5)
    for i in range(2, limit + 1):
        if spf[i] == i:
            step = i
            start = i * i
            for j in range(start, n + 1, step):
                if spf[j] == j:
                    spf[j] = i
    return spf


def _factor_unique_primes(n: int, spf: list[int]) -> list[int]:
    primes: list[int] = []
    while n > 1:
        p = spf[n]
        primes.append(p)
        while n % p == 0:
            n //= p
    return primes


def squarefree_divs_mu(
    n: int, spf: list[int], cache: dict[int, list[tuple[int, int, int, int]]]
) -> list[tuple[int, int, int, int]]:
    """Return squarefree divisors d of n together with μ(d), plus d and d^4.

    Output tuples are (d, mu, d, d4).
    Only squarefree divisors matter because μ(d)=0 otherwise.
    """
    if n in cache:
        return cache[n]

    if n <= 1:
        cache[n] = [(1, 1, 1, 1)]
        return cache[n]

    primes = _factor_unique_primes(n, spf)

    divs: list[tuple[int, int]] = [(1, 1)]
    for p in primes:
        divs += [(d * p, -mu) for (d, mu) in divs]

    out: list[tuple[int, int, int, int]] = []
    for d, mu in divs:
        d4 = d * d * d * d
        out.append((d, mu, d, d4))

    cache[n] = out
    return out


def coprime_odd_prefix(
    base: int,
    L: int,
    divs: list[tuple[int, int, int, int]],
    mod_s1: int,
    mod_s4: int,
) -> tuple[int, int, int]:
    """For fixed odd base, compute over odd t <= L with gcd(t, base)=1:

        count, sum(t) mod mod_s1, sum(t^4) mod mod_s4.

    Uses inclusion-exclusion over squarefree divisors of base.
    """
    if L <= 0:
        return 0, 0, 0

    cnt = 0
    s1 = 0
    s4 = 0

    for d, mu, d1, d4 in divs:
        m = L // d
        if m <= 0:
            continue

        cnt += mu * odd_count(m)

        if mod_s1:
            s1 = (s1 + mu * (d1 % mod_s1) * (odd_sum1(m) % mod_s1)) % mod_s1

        if mod_s4:
            s4 = (s4 + mu * (d4 % mod_s4) * (odd_sum4(m) % mod_s4)) % mod_s4

    return int(cnt), s1 % mod_s1, s4 % mod_s4


def coprime_odd_range(
    base: int,
    lo: int,
    hi: int,
    divs: list[tuple[int, int, int, int]],
    mod_s1: int,
    mod_s4: int,
) -> tuple[int, int, int]:
    """Same as coprime_odd_prefix, but restricted to odd t in [lo, hi]."""
    if hi < lo or hi <= 0:
        return 0, 0, 0
    if lo <= 1:
        return coprime_odd_prefix(base, hi, divs, mod_s1, mod_s4)

    c1, s1_1, s4_1 = coprime_odd_prefix(base, hi, divs, mod_s1, mod_s4)
    c0, s1_0, s4_0 = coprime_odd_prefix(base, lo - 1, divs, mod_s1, mod_s4)

    cnt = c1 - c0
    s1 = (s1_1 - s1_0) % mod_s1
    s4 = (s4_1 - s4_0) % mod_s4
    return cnt, s1, s4


def solve(
    N: int, mod: int = MOD_DEFAULT, want_count: bool = False
) -> int | tuple[int, int]:
    """Return S(N) modulo `mod`. If want_count=True, also return number of solutions."""

    MOD = mod
    MOD2 = 2 * MOD
    MOD8 = 8 * MOD

    # Upper bounds for the odd parameters are ~ (2N)^(1/4).
    max_base = iroot4(2 * N) + 2

    spf = sieve_spf(max_base)
    div_cache: dict[int, list[tuple[int, int, int, int]]] = {}

    # Precompute 4th powers for fast lookup.
    pow4 = [0] * (max_base + 1)
    for i in range(max_base + 1):
        pow4[i] = i * i * i * i

    total = 0
    total_count = 0

    # ----------------------------
    # Family A (primitive):
    #   z-4x = p^4, z+4x = q^4   with odd coprime p<q.
    # Then:
    #   x = (q^4 - p^4)/8,
    #   y = p*q,
    #   z = (p^4 + q^4)/2.
    # ----------------------------

    q_max = iroot4(2 * N - 1)
    for q in range(1, q_max + 1, 2):
        q4 = pow4[q]

        # Bounds for p
        rem = 2 * N - q4
        if rem <= 0:
            break

        p_max = iroot4(rem)  # from z <= N
        p_max = min(p_max, q - 1)  # from p < q (x>0)
        p_max = min(p_max, N // q)  # from y <= N

        if q4 <= 8:
            continue
        p_max = min(p_max, iroot4(q4 - 8))  # x >= 1
        if p_max <= 0:
            continue

        p_min = 1
        low = q4 - 8 * N  # from x <= N: p^4 >= q^4 - 8N
        if low > 1:
            p_min = ceil_root4(low)
        if p_min % 2 == 0:
            p_min += 1

        if p_min > p_max:
            continue

        divs = squarefree_divs_mu(q, spf, div_cache)
        cnt, s1, s4_mod8 = coprime_odd_range(q, p_min, p_max, divs, MOD, MOD8)
        if cnt <= 0:
            continue

        total_count += cnt

        # Sum_x = Σ (q^4 - p^4)/8
        # We only need it modulo MOD; since division by 8 is exact, compute modulo 8*MOD first.
        numx_mod8 = ((cnt % MOD8) * (q4 % MOD8) - s4_mod8) % MOD8
        sum_x = (numx_mod8 // 8) % MOD

        # Sum_z = Σ (p^4 + q^4)/2
        numz_mod2 = ((s4_mod8 % MOD2) + (cnt % MOD2) * (q4 % MOD2)) % MOD2
        sum_z = (numz_mod2 // 2) % MOD

        # Sum_y = Σ p*q
        sum_y = (q % MOD) * (s1 % MOD) % MOD

        total = (total + sum_x + sum_y + sum_z) % MOD

    # ----------------------------
    # Family B (primitive): min(v2(z-4x), v2(z+4x)) = 3.
    # Let k>=1 and 4k+1 be the other 2-adic exponent.
    # This yields two sub-families depending on which factor is larger.
    # ----------------------------

    k = 1
    while (1 << (4 * k)) <= N:
        scale4k = 1 << (4 * k)  # 2^(4k)
        scale4k_2 = 1 << (4 * k - 2)  # 2^(4k-2)
        y_scale = 1 << (k + 1)  # 2^(k+1)

        # Case B_high: z-4x = 8*p^4, z+4x = 2^(4k+1)*q^4
        # => x = 2^(4k-2)*q^4 - p^4
        #    y = 2^(k+1)*p*q
        #    z = 4*p^4 + 2^(4k)*q^4
        if N > 4:
            q_max2 = iroot4((N - 4) // scale4k)
            for q in range(1, q_max2 + 1, 2):
                q4 = pow4[q]

                remN = N - scale4k * q4
                if remN < 4:
                    continue

                p_max_z = iroot4(remN // 4)  # from z<=N

                bound_pos = scale4k_2 * q4 - 1  # from x>=1 and x>0
                if bound_pos <= 0:
                    continue
                p_max_pos = iroot4(bound_pos)

                p_max = min(p_max_z, p_max_pos, N // (y_scale * q))
                if p_max <= 0:
                    continue

                # from x<=N: p^4 >= 2^(4k-2)*q^4 - N
                low = scale4k_2 * q4 - N
                p_min = 1
                if low > 1:
                    p_min = ceil_root4(low)
                if p_min % 2 == 0:
                    p_min += 1

                if p_min > p_max:
                    continue

                divs = squarefree_divs_mu(q, spf, div_cache)
                cnt, s1, s4 = coprime_odd_range(q, p_min, p_max, divs, MOD, MOD)
                if cnt <= 0:
                    continue

                total_count += cnt

                # Σ(3*p^4 + 5*2^(4k-2)*q^4 + 2^(k+1)*p*q)
                term_p4 = (3 * s4) % MOD
                term_q4 = (5 * (scale4k_2 % MOD) * (q4 % MOD) * (cnt % MOD)) % MOD
                term_pq = ((y_scale % MOD) * (q % MOD) * (s1 % MOD)) % MOD
                total = (total + term_p4 + term_q4 + term_pq) % MOD

        # Case B_low: z-4x = 2^(4k+1)*p^4, z+4x = 8*q^4
        # => x = q^4 - 2^(4k-2)*p^4
        #    y = 2^(k+1)*p*q
        #    z = 4*q^4 + 2^(4k)*p^4
        p_max_all = iroot4(N // scale4k)
        for p in range(1, p_max_all + 1, 2):
            p4 = pow4[p]

            remN = N - scale4k * p4
            if remN < 4:
                continue

            q_max_z = iroot4(remN // 4)  # from z<=N
            q_max = min(q_max_z, N // (y_scale * p))

            # from x<=N: q^4 <= N + 2^(4k-2)*p^4
            q_max = min(q_max, iroot4(N + scale4k_2 * p4))
            if q_max <= 0:
                continue

            # from x>=1: q^4 >= 2^(4k-2)*p^4 + 1
            q_min = ceil_root4(scale4k_2 * p4 + 1)
            if q_min % 2 == 0:
                q_min += 1

            if q_min > q_max:
                continue

            divs = squarefree_divs_mu(p, spf, div_cache)
            cnt, s1, s4 = coprime_odd_range(p, q_min, q_max, divs, MOD, MOD)
            if cnt <= 0:
                continue

            total_count += cnt

            # Σ(5*q^4 + 3*2^(4k-2)*p^4 + 2^(k+1)*p*q)
            term_q4 = (5 * s4) % MOD
            term_p4 = (3 * (scale4k_2 % MOD) * (p4 % MOD) * (cnt % MOD)) % MOD
            term_pq = ((y_scale % MOD) * (p % MOD) * (s1 % MOD)) % MOD
            total = (total + term_q4 + term_p4 + term_pq) % MOD

        k += 1

    return (total, total_count) if want_count else total


def _self_test() -> None:
    # The two example solutions for N=100.
    x1, y1, z1 = 3, 4, 20
    x2, y2, z2 = 10, 3, 41
    assert 16 * x1 * x1 + y1**4 == z1 * z1
    assert 16 * x2 * x2 + y2**4 == z2 * z2
    assert gcd(gcd(x1, y1), z1) == 1
    assert gcd(gcd(x2, y2), z2) == 1

    s100, c100 = solve(10**2, want_count=True)
    assert c100 == 2
    assert s100 == 81

    s1e4, c1e4 = solve(10**4, want_count=True)
    assert c1e4 == 26
    assert s1e4 == 112851

    s1e7 = solve(10**7)
    assert s1e7 == 248876211


def main() -> None:
    _self_test()
    print(solve(10**16))


if __name__ == "__main__":
    main()
