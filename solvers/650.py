#!/usr/bin/env python
"""
Project Euler 650: Divisors of Binomial Product

Track the prime exponents of B_n = prod_k C(n,k) through the factorial identity
for B_n, and evaluate sigma(B_n) from those exponents modulo 1_000_000_007.
"""

from math import comb


LIMIT = 20_000
MOD = 1_000_000_007


def spf_and_primes(limit: int) -> tuple[list[int], list[int]]:
    spf = [0] * (limit + 1)
    primes: list[int] = []
    for x in range(2, limit + 1):
        if spf[x] == 0:
            spf[x] = x
            primes.append(x)
        spfx = spf[x]
        for p in primes:
            y = p * x
            if y > limit:
                break
            spf[y] = p
            if p == spfx:
                break
    return spf, primes


def factor_with_spf(n: int, spf: list[int]) -> list[tuple[int, int]]:
    factors: list[tuple[int, int]] = []
    while n > 1:
        p = spf[n]
        exponent = 0
        while n % p == 0:
            n //= p
            exponent += 1
        factors.append((p, exponent))
    return factors


def S(limit: int = LIMIT, mod: int = MOD) -> int:
    spf, primes = spf_and_primes(limit)
    prime_index = [0] * (limit + 1)
    for i, p in enumerate(primes):
        prime_index[p] = i

    prime_power = [p % mod for p in primes]  # p^(e_{n,p}+1), initially e=0.
    inv_factorial_power = [1] * len(primes)  # p^(-v_p(n!)).
    inv_prime = [pow(p, mod - 2, mod) for p in primes]
    sigma_den_inv = [pow(p - 1, mod - 2, mod) for p in primes]

    active = 0
    total = 1  # D_1

    for n in range(2, limit + 1):
        while active < len(primes) and primes[active] <= n:
            active += 1

        for i in range(active):
            prime_power[i] = prime_power[i] * inv_factorial_power[i] % mod

        for p, exponent in factor_with_spf(n, spf):
            i = prime_index[p]
            prime_power[i] = prime_power[i] * pow(p, (n - 1) * exponent, mod) % mod
            inv_factorial_power[i] = (
                inv_factorial_power[i] * pow(inv_prime[i], exponent, mod) % mod
            )

        divisor_sum = 1
        for i in range(active):
            divisor_sum = (
                divisor_sum
                * ((prime_power[i] - 1) % mod)
                % mod
                * sigma_den_inv[i]
                % mod
            )

        total = (total + divisor_sum) % mod

    return total


def B_small(n: int) -> int:
    total = 1
    for k in range(n + 1):
        total *= comb(n, k)
    return total


def prime_factors(n: int) -> dict[int, int]:
    factors: dict[int, int] = {}
    p = 2
    while p * p <= n:
        while n % p == 0:
            factors[p] = factors.get(p, 0) + 1
            n //= p
        p += 1 if p == 2 else 2
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors


def D_small(n: int) -> int:
    total = 1
    for p, exponent in prime_factors(B_small(n)).items():
        total *= (p ** (exponent + 1) - 1) // (p - 1)
    return total


def S_exact(n: int) -> int:
    return sum(D_small(k) for k in range(1, n + 1))


def main() -> None:
    assert B_small(5) == 2500
    assert D_small(5) == 5467
    assert S_exact(5) == 5736
    assert S_exact(10) == 141740594713218418
    assert S(100) == 332792866
    print(S())


if __name__ == "__main__":
    main()
