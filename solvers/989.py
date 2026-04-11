#!/usr/bin/env python
from __future__ import annotations

from array import array
from math import gcd, isqrt
from typing import List, Sequence, Tuple

MOD: int = 1_000_000_009
TARGET_LIMIT: int = 10**14
SMALL_NONPRIMITIVE_LIMIT: int = 8


def tonelli_shanks(n: int, p: int) -> int:
    if n == 0:
        return 0
    if pow(n, (p - 1) // 2, p) != 1:
        raise ValueError("not a quadratic residue")
    if p % 4 == 3:
        return pow(n, (p + 1) // 4, p)

    q: int = p - 1
    s: int = 0
    while q % 2 == 0:
        q //= 2
        s += 1

    z: int = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1

    m: int = s
    c: int = pow(z, q, p)
    t: int = pow(n, q, p)
    r: int = pow(n, (q + 1) // 2, p)

    while t != 1:
        i: int = 1
        t2i: int = t * t % p
        while t2i != 1:
            t2i = t2i * t2i % p
            i += 1
        b: int = pow(c, 1 << (m - i - 1), p)
        r = r * b % p
        c = b * b % p
        t = t * c % p
        m = i

    return r


SQRT5_MOD: int = tonelli_shanks(5, MOD)
INV_SQRT5_MOD: int = pow(SQRT5_MOD, MOD - 2, MOD)
INV2_MOD: int = (MOD + 1) // 2
PHI_MOD: int = (1 + SQRT5_MOD) * INV2_MOD % MOD
PHI_INV_MOD: int = pow(PHI_MOD, MOD - 2, MOD)
PSI_MOD: int = (1 - SQRT5_MOD) * INV2_MOD % MOD
PHI_SQUARED_MOD: int = PHI_MOD * PHI_MOD % MOD
PHI_INV_SQUARED_MOD: int = PHI_INV_MOD * PHI_INV_MOD % MOD


def build_small_nonprimitive_terms(max_limit: int) -> List[Tuple[int, ...]]:
    values: List[int] = []
    max_a: int = 2 * isqrt(max_limit) + 2
    for a in range(2, max_a + 1):
        for b in range(1, a // 2 + 1):
            q: int = a * a - a * b - b * b
            if 0 < q <= max_limit:
                values.append(q)
    values.sort()

    terms: List[Tuple[int, ...]] = [tuple() for _ in range(max_limit + 1)]
    prefix: List[int] = []
    index: int = 0
    total: int = len(values)
    for limit in range(max_limit + 1):
        while index < total and values[index] <= limit:
            prefix.append(values[index])
            index += 1
        terms[limit] = tuple(prefix)
    return terms


SMALL_NONPRIMITIVE_TERMS: List[Tuple[int, ...]] = build_small_nonprimitive_terms(
    SMALL_NONPRIMITIVE_LIMIT
)


def eval_small_nonprimitive_pair(limit: int, z1: int, z2: int) -> Tuple[int, int]:
    terms: Sequence[int] = SMALL_NONPRIMITIVE_TERMS[limit]
    total1: int = 0
    total2: int = 0
    power1: int = 1
    power2: int = 1
    exponent: int = 0

    for target in terms:
        while exponent < target:
            power1 = power1 * z1 % MOD
            power2 = power2 * z2 % MOD
            exponent += 1
        total1 += power1
        if total1 >= MOD:
            total1 -= MOD
        total2 += power2
        if total2 >= MOD:
            total2 -= MOD

    return total1, total2


def mobius_sieve(limit: int) -> array:
    mu: array = array("b", [1]) * (limit + 1)
    is_prime: bytearray = bytearray(b"\x01") * (limit + 1)
    if limit >= 0:
        is_prime[0] = 0
    if limit >= 1:
        is_prime[1] = 0

    for p in range(2, limit + 1):
        if not is_prime[p]:
            continue
        for multiple in range(p, limit + 1, p):
            mu[multiple] = -mu[multiple]
        square: int = p * p
        if square <= limit:
            for multiple in range(square, limit + 1, square):
                mu[multiple] = 0
            for multiple in range(square, limit + 1, p):
                is_prime[multiple] = 0
        for multiple in range(p + p, limit + 1, p):
            is_prime[multiple] = 0

    return mu


def nonprimitive_pair(
    limit: int, z1: int, z1_inv: int, z2: int, z2_inv: int
) -> Tuple[int, int]:
    if limit <= SMALL_NONPRIMITIVE_LIMIT:
        return eval_small_nonprimitive_pair(limit, z1, z2)

    mod: int = MOD
    total1: int = 0
    total2: int = 0

    z1_sq: int = z1 * z1 % mod
    z2_sq: int = z2 * z2 % mod

    z1_inv_sq: int = z1_inv * z1_inv % mod
    z2_inv_sq: int = z2_inv * z2_inv % mod
    z1_inv_4: int = z1_inv_sq * z1_inv_sq % mod
    z2_inv_4: int = z2_inv_sq * z2_inv_sq % mod
    z1_inv_5: int = z1_inv_4 * z1_inv % mod
    z2_inv_5: int = z2_inv_4 * z2_inv % mod
    z1_inv_10: int = z1_inv_5 * z1_inv_5 % mod
    z2_inv_10: int = z2_inv_5 * z2_inv_5 % mod
    z1_inv_15: int = z1_inv_10 * z1_inv_5 % mod
    z2_inv_15: int = z2_inv_10 * z2_inv_5 % mod

    # Maintain the moving window of z^(m^2) values for m in [lower, upper].
    even_weight1: int = z1_inv_5
    even_weight2: int = z2_inv_5
    even_delta1: int = z1_inv_15
    even_delta2: int = z2_inv_15

    add_index: int = 0
    add_term1: int = 1
    add_term2: int = 1
    add_step1: int = z1
    add_step2: int = z2

    drop_index: int = 0
    drop_term1: int = 1
    drop_term2: int = 1
    drop_step1: int = z1
    drop_step2: int = z2

    window1: int = 0
    window2: int = 0
    t: int = 1
    lower: int = 3
    upper: int = 0
    rhs: int = limit + 5

    while (upper + 1) * (upper + 1) <= rhs:
        upper += 1

    while lower <= upper:
        while add_index <= upper:
            window1 += add_term1
            if window1 >= mod:
                window1 -= mod
            window2 += add_term2
            if window2 >= mod:
                window2 -= mod

            add_term1 = add_term1 * add_step1 % mod
            add_step1 = add_step1 * z1_sq % mod
            add_term2 = add_term2 * add_step2 % mod
            add_step2 = add_step2 * z2_sq % mod
            add_index += 1

        while drop_index < lower:
            window1 -= drop_term1
            if window1 < 0:
                window1 += mod
            window2 -= drop_term2
            if window2 < 0:
                window2 += mod

            drop_term1 = drop_term1 * drop_step1 % mod
            drop_step1 = drop_step1 * z1_sq % mod
            drop_term2 = drop_term2 * drop_step2 % mod
            drop_step2 = drop_step2 * z2_sq % mod
            drop_index += 1

        total1 = (total1 + window1 * even_weight1) % mod
        total2 = (total2 + window2 * even_weight2) % mod

        even_weight1 = even_weight1 * even_delta1 % mod
        even_delta1 = even_delta1 * z1_inv_10 % mod
        even_weight2 = even_weight2 * even_delta2 % mod
        even_delta2 = even_delta2 * z2_inv_10 % mod

        rhs += 10 * t + 5
        t += 1
        lower += 3
        while (upper + 1) * (upper + 1) <= rhs:
            upper += 1

    # Same idea for z^(m(m+1)).
    odd_weight1: int = z1_inv
    odd_weight2: int = z2_inv
    odd_delta1: int = z1_inv_10
    odd_delta2: int = z2_inv_10

    add_index = 0
    add_term1 = 1
    add_term2 = 1
    add_step1 = z1_sq
    add_step2 = z2_sq

    drop_index = 0
    drop_term1 = 1
    drop_term2 = 1
    drop_step1 = z1_sq
    drop_step2 = z2_sq

    window1 = 0
    window2 = 0
    t = 0
    lower = 1
    upper = 0
    rhs = limit + 1

    while (upper + 1) * (upper + 2) <= rhs:
        upper += 1

    while lower <= upper:
        while add_index <= upper:
            window1 += add_term1
            if window1 >= mod:
                window1 -= mod
            window2 += add_term2
            if window2 >= mod:
                window2 -= mod

            add_term1 = add_term1 * add_step1 % mod
            add_step1 = add_step1 * z1_sq % mod
            add_term2 = add_term2 * add_step2 % mod
            add_step2 = add_step2 * z2_sq % mod
            add_index += 1

        while drop_index < lower:
            window1 -= drop_term1
            if window1 < 0:
                window1 += mod
            window2 -= drop_term2
            if window2 < 0:
                window2 += mod

            drop_term1 = drop_term1 * drop_step1 % mod
            drop_step1 = drop_step1 * z1_sq % mod
            drop_term2 = drop_term2 * drop_step2 % mod
            drop_step2 = drop_step2 * z2_sq % mod
            drop_index += 1

        total1 = (total1 + window1 * odd_weight1) % mod
        total2 = (total2 + window2 * odd_weight2) % mod

        odd_weight1 = odd_weight1 * odd_delta1 % mod
        odd_delta1 = odd_delta1 * z1_inv_10 % mod
        odd_weight2 = odd_weight2 * odd_delta2 % mod
        odd_delta2 = odd_delta2 * z2_inv_10 % mod

        rhs += 10 * t + 10
        t += 1
        lower += 3
        while (upper + 1) * (upper + 2) <= rhs:
            upper += 1

    return total1, total2


def solve(limit: int) -> int:
    root: int = isqrt(limit)
    mu: array = mobius_sieve(root)

    p_phi: int = 0
    p_psi: int = 0

    phi_pow_g2: int = 1
    phi_inv_pow_g2: int = 1
    forward_step: int = PHI_MOD
    backward_step: int = PHI_INV_MOD
    g_square: int = 1

    for g in range(1, root + 1):
        phi_pow_g2 = phi_pow_g2 * forward_step % MOD
        forward_step = forward_step * PHI_SQUARED_MOD % MOD
        phi_inv_pow_g2 = phi_inv_pow_g2 * backward_step % MOD
        backward_step = backward_step * PHI_INV_SQUARED_MOD % MOD

        mu_g: int = mu[g]
        if mu_g:
            scaled_limit: int = limit // g_square
            if g & 1:
                psi_pow_g2: int = MOD - phi_inv_pow_g2
                psi_inv_pow_g2: int = MOD - phi_pow_g2
            else:
                psi_pow_g2 = phi_inv_pow_g2
                psi_inv_pow_g2 = phi_pow_g2

            nonprimitive_phi, nonprimitive_psi = nonprimitive_pair(
                scaled_limit,
                phi_pow_g2,
                phi_inv_pow_g2,
                psi_pow_g2,
                psi_inv_pow_g2,
            )

            if mu_g == 1:
                p_phi += nonprimitive_phi
                if p_phi >= MOD:
                    p_phi -= MOD
                p_psi += nonprimitive_psi
                if p_psi >= MOD:
                    p_psi -= MOD
            else:
                p_phi -= nonprimitive_phi
                if p_phi < 0:
                    p_phi += MOD
                p_psi -= nonprimitive_psi
                if p_psi < 0:
                    p_psi += MOD

        g_square += 2 * g + 1

    return (p_phi - p_psi) % MOD * INV_SQRT5_MOD % MOD


def brute_g(n: int) -> int:
    count: int = 0
    for x in range(n):
        if (x * x - x - 1) % n == 0:
            count += 1
    return count


def factorize_small(n: int) -> List[Tuple[int, int]]:
    factors: List[Tuple[int, int]] = []
    d: int = 2
    while d * d <= n:
        if n % d == 0:
            exponent: int = 0
            while n % d == 0:
                n //= d
                exponent += 1
            factors.append((d, exponent))
        d += 1 if d == 2 else 2
    if n > 1:
        factors.append((n, 1))
    return factors


def g_from_factorization(n: int) -> int:
    if n == 1:
        return 1

    split_prime_count: int = 0
    for prime, exponent in factorize_small(n):
        if prime == 2:
            return 0
        if prime == 5:
            if exponent >= 2:
                return 0
            continue
        residue: int = prime % 5
        if residue in (2, 3):
            return 0
        split_prime_count += 1

    return 1 << split_prime_count


def reduced_pair_count(n: int) -> int:
    count: int = 0
    max_a: int = 2 * isqrt(n) + 2
    for a in range(2, max_a + 1):
        for b in range(1, a // 2 + 1):
            if gcd(a, b) != 1:
                continue
            if a * a - a * b - b * b == n:
                count += 1
    return count


def brute_nonprimitive_pair(limit: int, z1: int, z2: int) -> Tuple[int, int]:
    total1: int = 0
    total2: int = 0
    max_a: int = 2 * isqrt(limit) + 2
    for a in range(2, max_a + 1):
        for b in range(1, a // 2 + 1):
            q: int = a * a - a * b - b * b
            if q <= limit:
                total1 = (total1 + pow(z1, q, MOD)) % MOD
                total2 = (total2 + pow(z2, q, MOD)) % MOD
    return total1, total2


def brute_fibonacci_sum(limit: int) -> int:
    fib: List[int] = [0] * (limit + 1)
    if limit >= 1:
        fib[1] = 1
    if limit >= 2:
        fib[2] = 1
    for n in range(3, limit + 1):
        fib[n] = (fib[n - 1] + fib[n - 2]) % MOD

    total: int = 0
    for n in range(1, limit + 1):
        total = (total + fib[n] * brute_g(n)) % MOD
    return total


def validate() -> None:
    assert PSI_MOD == (MOD - PHI_INV_MOD) % MOD

    for n in range(1, 200):
        brute: int = brute_g(n)
        factorized: int = g_from_factorization(n)
        reduced: int = reduced_pair_count(n)
        assert brute == factorized == reduced

    for limit in range(SMALL_NONPRIMITIVE_LIMIT + 1):
        fast_pair: Tuple[int, int] = eval_small_nonprimitive_pair(limit, 2, 3)
        brute_pair: Tuple[int, int] = brute_nonprimitive_pair(limit, 2, 3)
        assert fast_pair == brute_pair

    for limit in (1, 2, 5, 10, 30, 100):
        assert solve(limit) == brute_fibonacci_sum(limit)

    assert solve(1_000) == 190_950_976


def main() -> None:
    validate()
    print(solve(TARGET_LIMIT))


if __name__ == "__main__":
    main()
