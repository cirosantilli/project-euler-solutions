#!/usr/bin/env python
"""Project Euler 342 - The Totient of a Square Is a Cube."""

from __future__ import annotations

import math
import sys
from bisect import bisect_right


LIMIT_N = 10**10


def build_spf(limit: int) -> list[int]:
    spf = list(range(limit + 1))
    if limit >= 1:
        spf[1] = 0

    for p in range(2, math.isqrt(limit) + 1):
        if spf[p] != p:
            continue
        for multiple in range(p * p, limit + 1, p):
            if spf[multiple] == multiple:
                spf[multiple] = p

    return spf


def phi(n: int) -> int:
    result = n
    x = n
    p = 2
    while p * p <= x:
        if x % p == 0:
            while x % p == 0:
                x //= p
            result -= result // p
        p = 3 if p == 2 else p + 2
    if x > 1:
        result -= result // x
    return result


def is_perfect_cube(x: int) -> bool:
    root = round(x ** (1.0 / 3.0))
    return any(candidate >= 0 and candidate**3 == x for candidate in range(root - 1, root + 2))


def p_minus_1_factors_mod3(p: int, spf: list[int]) -> tuple[tuple[int, int], ...]:
    x = p - 1
    factors: list[tuple[int, int]] = []

    while x > 1:
        q = spf[x]
        exponent = 0
        while x % q == 0:
            x //= q
            exponent += 1
        exponent %= 3
        if exponent:
            factors.append((q, exponent))

    return tuple(factors)


def solve(limit_n: int = LIMIT_N) -> int:
    max_prime = math.isqrt(limit_n - 1)
    spf = build_spf(max_prime)
    primes = [p for p in range(2, max_prime + 1) if spf[p] == p]
    prime_index = {p: i for i, p in enumerate(primes)}
    factors_mod3 = [()] * (max_prime + 1)
    for p in primes:
        factors_mod3[p] = p_minus_1_factors_mod3(p, spf)

    residues = [0] * (max_prime + 1)
    nonzero_primes: set[int] = set()
    selected_primes: list[int] = []
    total = 0

    sys.setrecursionlimit(len(primes) + 100)

    def add_residue(p: int, delta: int) -> None:
        old = residues[p]
        new = (old + delta) % 3
        if old == 0 and new:
            nonzero_primes.add(p)
        elif old and new == 0:
            nonzero_primes.remove(p)
        residues[p] = new

    def add_factor_residues(p: int, sign: int) -> None:
        for q, exponent in factors_mod3[p]:
            add_residue(q, sign * exponent)

    def multiplier_sum(index: int, current: int, max_multiplier: int) -> int:
        if index == len(selected_primes):
            return current

        p3 = selected_primes[index] ** 3
        subtotal = 0
        value = current
        while value <= max_multiplier:
            subtotal += multiplier_sum(index + 1, value, max_multiplier)
            value *= p3
        return subtotal

    def add_family(base: int) -> None:
        nonlocal total

        if base > 1:
            total += base * multiplier_sum(0, 1, (limit_n - 1) // base)

    def include_optional(index: int, base: int) -> None:
        p = primes[index]
        selected_primes.append(p)
        add_factor_residues(p, 1)
        dfs(index - 1, base * p * p)
        add_factor_residues(p, -1)
        selected_primes.pop()

    def force_prime(index: int, base: int) -> None:
        p = primes[index]
        residue = residues[p]
        exponent = 3 if residue == 1 else 1
        next_base = base * (p**exponent)
        if next_base >= limit_n:
            return

        add_residue(p, -residue)
        selected_primes.append(p)
        add_factor_residues(p, 1)
        dfs(index - 1, next_base)
        add_factor_residues(p, -1)
        selected_primes.pop()
        add_residue(p, residue)

    def dfs(high_index: int, base: int) -> None:
        if high_index < 0:
            if not nonzero_primes:
                add_family(base)
            return

        max_optional = math.isqrt((limit_n - 1) // base)
        optional_end = min(high_index, bisect_right(primes, max_optional) - 1)

        if nonzero_primes:
            forced_prime = max(nonzero_primes)
            forced_index = prime_index[forced_prime]

            for index in range(optional_end, forced_index, -1):
                include_optional(index, base)
            force_prime(forced_index, base)
            return

        add_family(base)
        for index in range(optional_end, -1, -1):
            include_optional(index, base)

    dfs(len(primes) - 1, 1)
    return total


def main() -> None:
    # Test value from the problem statement:
    # n = 50 => phi(50^2) = phi(2500) = 2^3 * 5^3 = 1000, which is a cube.
    assert is_perfect_cube(phi(50 * 50))

    print(solve())


if __name__ == "__main__":
    main()
