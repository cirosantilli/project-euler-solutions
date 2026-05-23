#!/usr/bin/env python
"""
Project Euler 296: Angular Bisector and Tangent

The geometry gives BE = a*c/(a+b).  Writing a = g*x, b = g*(q-x),
gcd(x, q) = 1, and c = k*q turns the count into sums over reduced residues.
Those sums are evaluated by Mobius inversion and floor-sum blocks.
"""

from math import gcd


LIMIT = 100_000


def brute_force(limit: int) -> int:
    count = 0
    for a in range(1, limit // 3 + 1):
        for b in range(a, (limit - a) // 2 + 1):
            for c in range(b, min(a + b, limit - a - b + 1)):
                if a * c % (a + b) == 0:
                    count += 1
    return count


def pair_count(limit: int) -> int:
    count = 0
    gcd_ = gcd

    for a in range(1, limit // 3 + 1):
        for b in range(a, (limit - a) // 2 + 1):
            c_max = min(a + b - 1, limit - a - b)
            step = (a + b) // gcd_(a, b)
            count += c_max // step - (b - 1) // step

    return count


def floor_sum(n: int, m: int, a: int, b: int = 0) -> int:
    """Return sum_{0 <= i < n} floor(a*i/m)."""
    total = 0
    while True:
        if a >= m:
            total += (n - 1) * n * (a // m) // 2
            a %= m
        if b >= m:
            total += n * (b // m)
            b %= m

        y = a * n + b
        if y < m:
            return total

        n = y // m
        b = y % m
        m, a = a, m


def mobius_sieve(limit: int) -> list[int]:
    mu = [0] * (limit + 1)
    mu[1] = 1
    primes: list[int] = []
    composite = bytearray(limit + 1)

    for n in range(2, limit + 1):
        if not composite[n]:
            primes.append(n)
            mu[n] = -1
        for p in primes:
            value = n * p
            if value > limit:
                break
            composite[value] = 1
            if n % p == 0:
                mu[value] = 0
                break
            mu[value] = -mu[n]

    return mu


def divisor_mobius(limit: int) -> list[list[tuple[int, int]]]:
    mu = mobius_sieve(limit)
    divisors: list[list[tuple[int, int]]] = [[] for _ in range(limit + 1)]
    for d in range(1, limit + 1):
        if mu[d] == 0:
            continue
        for multiple in range(d, limit + 1, d):
            divisors[multiple].append((d, mu[d]))
    return divisors


def coprime_count(x: int, divisors: list[tuple[int, int]]) -> int:
    if x <= 0:
        return 0
    return sum(mu * (x // d) for d, mu in divisors)


def coprime_floor_sum(q: int, g: int, x: int, divisors: list[tuple[int, int]]) -> int:
    if x <= 0:
        return 0

    total = 0
    for d, mu in divisors:
        n = x // d
        if n:
            total += mu * floor_sum(n + 1, q // d, g)
    return total


def count_triangles(limit: int = LIMIT) -> int:
    max_q = limit // 3
    divisors_by_q = divisor_mobius(max_q)
    count = 0

    for q in range(2, max_q + 1):
        max_g_plus_k = limit // q
        half_q = q // 2
        divisors = divisors_by_q[q]
        reduced_up_to_half = coprime_count(half_q, divisors)

        for g in range(2, max_g_plus_k):
            min_h = max(1, 2 * g - max_g_plus_k)
            before = (min_h * q + g - 1) // g - 1
            if before >= half_q:
                continue

            reduced_count = reduced_up_to_half - coprime_count(before, divisors)
            floor_part = coprime_floor_sum(
                q, g, half_q, divisors
            ) - coprime_floor_sum(q, g, before, divisors)
            count += floor_part - (min_h - 1) * reduced_count

    return count


def main() -> None:
    assert count_triangles(150) == brute_force(150)
    assert count_triangles(300) == brute_force(300)
    assert count_triangles(1000) == pair_count(1000)
    print(count_triangles())


if __name__ == "__main__":
    main()
