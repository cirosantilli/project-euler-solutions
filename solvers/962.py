#!/usr/bin/env python
from __future__ import annotations

from math import gcd, isqrt


def integer_cuberoot_floor(n: int) -> int:
    x = int(round(n ** (1.0 / 3.0)))
    while (x + 1) ** 3 <= n:
        x += 1
    while x**3 > n:
        x -= 1
    return x


def integer_sqrt_ceiling(n: int) -> int:
    root = isqrt(n)
    return root if root * root == n else root + 1


def smallest_prime_factors(limit: int) -> list[int]:
    spf = list(range(limit + 1))
    if limit >= 1:
        spf[1] = 1
    for p in range(2, isqrt(limit) + 1):
        if spf[p] == p:
            start = p * p
            for multiple in range(start, limit + 1, p):
                if spf[multiple] == multiple:
                    spf[multiple] = p
    return spf


def factor_table(limit: int) -> list[tuple[tuple[int, int], ...]]:
    spf = smallest_prime_factors(limit)
    factors: list[tuple[tuple[int, int], ...]] = [()] * (limit + 1)
    for n in range(2, limit + 1):
        p = spf[n]
        m = n // p
        exponent = 1
        while m > 1 and spf[m] == p:
            m //= p
            exponent += 1
        factors[n] = ((p, exponent),) + factors[m]
    return factors


def squarefree_parts(
    factors: list[tuple[tuple[int, int], ...]], limit: int
) -> list[int]:
    parts = [1] * (limit + 1)
    for n in range(2, limit + 1):
        part = 1
        for prime, exponent in factors[n]:
            if exponent & 1:
                part *= prime
        parts[n] = part
    return parts


def minimal_square_base(
    d_factors: tuple[tuple[int, int], ...],
    r_factors: tuple[tuple[int, int], ...],
    v_factors: tuple[tuple[int, int], ...],
    limit: int,
) -> int:
    i = j = k = 0
    base = 1
    while i < len(d_factors) or j < len(r_factors) or k < len(v_factors):
        next_prime = 10**18
        if i < len(d_factors) and d_factors[i][0] < next_prime:
            next_prime = d_factors[i][0]
        if j < len(r_factors) and r_factors[j][0] < next_prime:
            next_prime = r_factors[j][0]
        if k < len(v_factors) and v_factors[k][0] < next_prime:
            next_prime = v_factors[k][0]

        exponent = 0
        if i < len(d_factors) and d_factors[i][0] == next_prime:
            exponent += d_factors[i][1]
            i += 1
        if j < len(r_factors) and r_factors[j][0] == next_prime:
            exponent += r_factors[j][1]
            j += 1
        if k < len(v_factors) and v_factors[k][0] == next_prime:
            exponent -= v_factors[k][1]
            k += 1

        if exponent < 0:
            base *= next_prime ** (-exponent)
        elif exponent & 1:
            base *= next_prime
        if base > limit:
            return base
    return base


def count_with_parity(lo: int, hi: int, parity: int) -> int:
    if hi < lo:
        return 0
    if (lo & 1) != parity:
        lo += 1
    if lo > hi:
        return 0
    return (hi - lo) // 2 + 1


def count_triangles(limit: int) -> int:
    k_limit = integer_cuberoot_floor(2 * limit * limit) + 2
    factor_limit = max(limit // 3 + 1, k_limit + 1)
    factors = factor_table(factor_limit)
    squarefree = squarefree_parts(factors, factor_limit)

    total = 0
    for k in range(2, k_limit + 1):
        max_s = limit // k
        if max_s == 0:
            break
        max_product = max_s * max_s
        for v in range((k + 1) // 2, k):
            if gcd(v, k) != 1:
                continue
            u = k - v
            d = squarefree[u]
            if v * d > max_product:
                continue

            r_limit = u * max_s // (u + 2 * v)
            if r_limit <= 0:
                continue

            common_dv = gcd(d, v)
            d_reduced = d // common_dv
            v_reduced = v // common_dv
            numerator = u + 2 * v
            for r in range(1, r_limit + 1):
                common_rv = gcd(r, v_reduced)
                remaining_r = r // common_rv
                numerator_kernel = squarefree[remaining_r]
                common = gcd(d_reduced, numerator_kernel)
                base = (v_reduced // common_rv) * (
                    d_reduced * numerator_kernel // (common * common)
                )
                if base > max_s:
                    continue

                s_min = (numerator * r + u - 1) // u
                n_min = integer_sqrt_ceiling((s_min + base - 1) // base)
                n_max = isqrt(max_s // base)
                if base & 1:
                    total += count_with_parity(n_min, n_max, r & 1)
                elif r & 1 == 0:
                    total += max(0, n_max - n_min + 1)
    return total


def brute_force(limit: int) -> int:
    count = 0
    for a in range(1, limit // 3 + 1):
        for b in range(a, (limit - a) // 2 + 1):
            max_c = min(a + b - 1, limit - a - b)
            for c in range(b, max_c + 1):
                numerator = a**3 * (a + b - c) * (a + b + c)
                denominator = b * (a + b) ** 2
                if numerator % denominator != 0:
                    continue
                q = numerator // denominator
                root = isqrt(q)
                if root * root == q:
                    count += 1
    return count


def main() -> None:
    for test_limit in (60, 80, 100, 150, 200):
        assert count_triangles(test_limit) == brute_force(test_limit)
    print(count_triangles(10**6))


if __name__ == "__main__":
    main()
