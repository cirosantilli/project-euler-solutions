#!/usr/bin/env python
from __future__ import annotations

"""
Project Euler 474: Last Digits of Divisors

For the main case, the required decimal suffix fixes the exact powers of 2
and 5 in the divisor.  The remaining unit part is counted with a dynamic
program over the unit residues modulo the reduced power of 10.
"""

from array import array
from math import gcd, isqrt


MOD = 10**16 + 61


def sieve_primes(limit: int) -> list[int]:
    if limit < 2:
        return []
    flags = bytearray(b"\x01") * (limit + 1)
    flags[0:2] = b"\x00\x00"
    for p in range(2, isqrt(limit) + 1):
        if flags[p]:
            start = p * p
            flags[start : limit + 1 : p] = b"\x00" * (((limit - start) // p) + 1)
    return [p for p in range(2, limit + 1) if flags[p]]


def exponent_in_factorial(n: int, p: int) -> int:
    exponent = 0
    while n:
        n //= p
        exponent += n
    return exponent


def v_factor(n: int, p: int) -> int:
    exponent = 0
    while n % p == 0:
        n //= p
        exponent += 1
    return exponent


def brute_count(n: int, d: int) -> int:
    """Small fallback for checks where the suffix is not a unit case."""
    modulus = 10 ** len(str(d))
    residues = {1: 1}
    for p in sieve_primes(n):
        powers = [1]
        step = p % modulus
        for _ in range(exponent_in_factorial(n, p)):
            powers.append(powers[-1] * step % modulus)

        new: dict[int, int] = {}
        for residue, count in residues.items():
            for power in powers:
                nxt = residue * power % modulus
                new[nxt] = (new.get(nxt, 0) + count) % MOD
        residues = new
    return residues.get(d % modulus, 0)


class UnitResidueDp:
    def __init__(self, modulus: int) -> None:
        self.modulus = modulus
        self.units = [r for r in range(1, modulus) if gcd(r, modulus) == 1]
        self.index = [-1] * modulus
        for i, residue in enumerate(self.units):
            self.index[residue] = i
        self.cycles: dict[int, list[array[int]]] = {}

    def cycles_for(self, multiplier: int) -> list[array[int]]:
        multiplier %= self.modulus
        cached = self.cycles.get(multiplier)
        if cached is not None:
            return cached

        seen = bytearray(len(self.units))
        cycles: list[array[int]] = []
        for start in range(len(self.units)):
            if seen[start]:
                continue
            cycle = array("H")
            idx = start
            while not seen[idx]:
                seen[idx] = 1
                cycle.append(idx)
                idx = self.index[self.units[idx] * multiplier % self.modulus]
            cycles.append(cycle)

        self.cycles[multiplier] = cycles
        return cycles

    def apply_prime(self, dp: list[int], multiplier: int, terms: int) -> list[int]:
        """Apply the transition sum_{j=0}^{terms-1} multiplier^j."""
        new = [0] * len(dp)
        for cycle in self.cycles_for(multiplier):
            length = len(cycle)
            full_turns, tail = divmod(terms, length)
            values = [dp[idx] for idx in cycle]
            full = (full_turns % MOD) * (sum(values) % MOD) % MOD

            if tail == 0:
                for idx in cycle:
                    new[idx] = full
                continue

            window = sum(values[-j % length] for j in range(tail)) % MOD
            for i, idx in enumerate(cycle):
                new[idx] = (full + window) % MOD
                window += values[(i + 1) % length]
                window -= values[(i + 1 - tail) % length]
                window %= MOD

        return new


def count_F_factorial(n: int, d: int) -> int:
    digits = len(str(d))
    alpha = v_factor(d, 2)
    beta = v_factor(d, 5)

    # The main reduction requires the target suffix to have exact 2- and
    # 5-adic valuations below the decimal modulus.  The only repository check
    # outside this regime is tiny, so direct residue DP is enough there.
    if alpha >= digits or beta >= digits:
        return brute_count(n, d)

    if alpha > exponent_in_factorial(n, 2) or beta > exponent_in_factorial(n, 5):
        return 0

    fixed = (2**alpha) * (5**beta)
    modulus = 10**digits // fixed
    target = d // fixed
    if gcd(target, modulus) != 1:
        return 0

    primes = sieve_primes(n)
    unit_dp = UnitResidueDp(modulus)
    dp = [0] * len(unit_dp.units)
    dp[unit_dp.index[1]] = 1

    for p in primes:
        if p == 2 or p == 5:
            continue
        terms = exponent_in_factorial(n, p) + 1
        dp = unit_dp.apply_prime(dp, p, terms)

    return dp[unit_dp.index[target % modulus]] % MOD


def main() -> None:
    assert count_F_factorial(12, 12) == 11
    assert count_F_factorial(50, 123) == 17888
    print(count_F_factorial(10**6, 65432))


if __name__ == "__main__":
    main()
