#!/usr/bin/env python

from math import isqrt

MOD = 10**9 + 7
INV6 = pow(6, MOD - 2, MOD)


def prime_sieve(limit):
    flags = bytearray(b"\x01") * (limit + 1)
    flags[0:2] = b"\x00\x00"
    for p in range(2, isqrt(limit) + 1):
        if flags[p]:
            start = p * p
            flags[start : limit + 1 : p] = b"\x00" * (((limit - start) // p) + 1)
    odd_primes = [p for p in range(3, limit + 1, 2) if flags[p]]
    return flags, odd_primes


def sum_sq(n):
    n %= MOD
    return n * (n + 1) % MOD * (2 * n + 1) % MOD * INV6 % MOD


def contribution(kernel, bound, multiplier):
    t = isqrt(bound // kernel)
    return multiplier * (kernel % MOD) % MOD * sum_sq(t) % MOD


def can_complete(prod, last, odd_count, bound):
    first = last + 2
    if odd_count:
        return prod * first <= bound
    return prod * first * (first + 2) <= bound


def branch_sum(bound, target, multiplier, flags, odd_primes):
    if bound <= 0:
        return 0

    total = contribution(1, bound, multiplier) if target == 0 else 0
    stack = []
    plen = len(odd_primes)

    for idx, p in enumerate(odd_primes):
        if not can_complete(p, p, 1, bound):
            break
        stack.append((idx + 1, p, p, p, 1))

    flags_local = flags
    primes_local = odd_primes
    limit = len(flags_local) - 1
    bound_local = bound
    target_local = target
    multiplier_local = multiplier

    while stack:
        next_idx, prod, xor_val, last, odd_count = stack.pop()

        if odd_count:
            cand = target_local ^ xor_val
            if (
                cand > last
                and cand <= limit
                and flags_local[cand]
                and prod * cand <= bound_local
            ):
                total = (
                    total + contribution(prod * cand, bound_local, multiplier_local)
                ) % MOD

        next_odd_count = odd_count ^ 1
        for j in range(next_idx, plen):
            q = primes_local[j]
            new_prod = prod * q
            if not can_complete(new_prod, q, next_odd_count, bound_local):
                break
            stack.append((j + 1, new_prod, xor_val ^ q, q, next_odd_count))

    return total % MOD


def compute_S(n):
    prime_limit = isqrt(2 * n) + 10
    flags, odd_primes = prime_sieve(prime_limit)
    return (
        branch_sum(n, 0, 1, flags, odd_primes)
        + branch_sum(n // 2, 2, 2, flags, odd_primes)
    ) % MOD


if __name__ == "__main__":
    assert compute_S(10) == 14
    assert compute_S(100) == 455
    print(compute_S(10**14))
