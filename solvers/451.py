#!/usr/bin/env python
"""
Project Euler 451: Modular Inverses

Self-inverse residues modulo n are the solutions of x^2 == 1 (mod n).  Build
the full root set from prime-power roots using CRT projectors, then take the
largest root below the trivial root n - 1.
"""

from array import array


LIMIT = 20_000_000


def smallest_prime_factors(n: int) -> array:
    spf = array("I", [0]) * (n + 1)
    primes: list[int] = []
    for x in range(2, n + 1):
        if spf[x] == 0:
            spf[x] = x
            primes.append(x)
        spfx = spf[x]
        for p in primes:
            y = p * x
            if y > n:
                break
            spf[y] = p
            if p == spfx:
                break
    return spf


def prime_power_factors(n: int, spf: array) -> list[tuple[int, int, int]]:
    factors: list[tuple[int, int, int]] = []
    while n > 1:
        p = spf[n]
        exponent = 0
        prime_power = 1
        while n % p == 0:
            n //= p
            exponent += 1
            prime_power *= p
        factors.append((p, exponent, prime_power))
    return factors


def nonzero_root_offsets_from_minus_one(p: int, exponent: int) -> tuple[int, ...]:
    if p == 2:
        if exponent == 1:
            return ()
        if exponent == 2:
            return (2,)
        half = 1 << (exponent - 1)
        return (2, half, half + 2)
    return (2,)


def I(n: int, spf: array) -> int:
    roots = [n - 1]

    for p, exponent, prime_power in prime_power_factors(n, spf):
        offsets = nonzero_root_offsets_from_minus_one(p, exponent)
        if not offsets:
            continue

        cofactor = n // prime_power
        projector = cofactor * pow(cofactor, -1, prime_power) % n
        previous_roots = roots
        roots = previous_roots[:]
        for offset in offsets:
            delta = offset * projector % n
            roots.extend((root + delta) % n for root in previous_roots)

    excluded = n - 1
    best = 1
    for root in roots:
        if best < root < excluded:
            best = root
    return best


def compute(limit: int = LIMIT) -> int:
    spf = smallest_prime_factors(limit)
    return sum(I(n, spf) for n in range(3, limit + 1))


def main() -> None:
    spf = smallest_prime_factors(100)
    assert I(7, spf) == 1
    assert I(15, spf) == 11
    assert I(100, spf) == 51
    print(compute())


if __name__ == "__main__":
    main()
