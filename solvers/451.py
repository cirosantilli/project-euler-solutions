#!/usr/bin/env python
"""
Project Euler 451: Modular Inverses

Self-inverse residues modulo n are the solutions of x^2 == 1 (mod n).  Build
the full root set from prime-power roots using CRT projectors, then take the
largest root below the trivial root n - 1.
"""

LIMIT = 20_000_000


def smallest_prime_factors(n: int) -> list[int]:
    spf = list(range(n + 1))
    root = int(n**0.5)
    for p in range(2, root + 1):
        if spf[p] != p:
            continue
        for multiple in range(p * p, n + 1, p):
            if spf[multiple] == multiple:
                spf[multiple] = p
    return spf


def modular_inverse(a: int, modulus: int) -> int:
    b = modulus
    x0 = 1
    x1 = 0
    while b:
        q = a // b
        a, b = b, a - q * b
        x0, x1 = x1, x0 - q * x1
    return x0 % modulus


def I(n: int, spf: list[int]) -> int:
    roots = [n - 1]
    best = 1
    remaining = n

    while remaining > 1:
        p = spf[remaining]
        prime_power = 1
        while remaining % p == 0:
            remaining //= p
            prime_power *= p

        if prime_power == 2:
            continue

        cofactor = n // prime_power
        projector = cofactor * modular_inverse(cofactor % prime_power, prime_power) % n
        delta_two = (2 * projector) % n

        delta_half = 0
        delta_half_plus_two = 0
        has_extra_two_roots = prime_power % 2 == 0 and prime_power >= 8
        if has_extra_two_roots:
            half = prime_power // 2
            delta_half = (half * projector) % n
            delta_half_plus_two = ((half + 2) * projector) % n

        previous_roots = list(roots)
        for root in previous_roots:
            candidate = (root + delta_two) % n
            roots.append(candidate)
            if best < candidate < n - 1:
                best = candidate

            if has_extra_two_roots:
                candidate = (root + delta_half) % n
                roots.append(candidate)
                if best < candidate < n - 1:
                    best = candidate

                candidate = (root + delta_half_plus_two) % n
                roots.append(candidate)
                if best < candidate < n - 1:
                    best = candidate

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
