#!/usr/bin/env python
"""
Project Euler 675: 2^omega(n)

For n = prod p^e, the divisor sum S(n) = sum_{d|n} 2^omega(d) factors as
prod_p (2e + 1).  Along the factorial sequence, only the primes dividing the
next multiplier i change, so S(i!) can be updated by replacing those local
factors.
"""

from array import array


LIMIT = 10_000_000
MOD = 1_000_000_087


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


def v2_factorial(n: int) -> int:
    total = 0
    power = 2
    while power <= n:
        total += n // power
        power *= 2
    return total


def inverse_table(limit: int, mod: int) -> array:
    inv = array("I", [0]) * (limit + 1)
    inv[1] = 1
    for x in range(2, limit + 1):
        inv[x] = (mod - (mod // x) * inv[mod % x] % mod) % mod
    return inv


def F(n: int = LIMIT, mod: int = MOD) -> int:
    spf = smallest_prime_factors(n)
    exponents = array("I", [0]) * (n + 1)
    inverses = inverse_table(2 * v2_factorial(n) + 1, mod)

    running = 1
    total = 0

    for i in range(2, n + 1):
        x = i
        while x > 1:
            p = spf[x]
            count = 0
            while x % p == 0:
                x //= p
                count += 1

            old_exp = exponents[p]
            new_exp = old_exp + count
            old_factor = 2 * old_exp + 1
            new_factor = 2 * new_exp + 1
            running = running * new_factor % mod * inverses[old_factor] % mod
            exponents[p] = new_exp

        total += running
        total %= mod

    return total


def F_brutish(n: int, mod: int = MOD) -> int:
    exponents: dict[int, int] = {}
    running = 1
    total = 0
    for i in range(2, n + 1):
        x = i
        p = 2
        while p * p <= x:
            if x % p == 0:
                count = 0
                while x % p == 0:
                    x //= p
                    count += 1
                old_exp = exponents.get(p, 0)
                new_exp = old_exp + count
                running = (
                    running * (2 * new_exp + 1) * pow(2 * old_exp + 1, -1, mod)
                ) % mod
                exponents[p] = new_exp
            p += 1 if p == 2 else 2
        if x > 1:
            old_exp = exponents.get(x, 0)
            new_exp = old_exp + 1
            running = (
                running * (2 * new_exp + 1) * pow(2 * old_exp + 1, -1, mod)
            ) % mod
            exponents[x] = new_exp
        total = (total + running) % mod
    return total


def main() -> None:
    assert F(5) == 96
    assert F(50) == F_brutish(50)
    print(F())


if __name__ == "__main__":
    main()
