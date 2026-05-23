#!/usr/bin/env python
from __future__ import annotations

import bisect
import math


def triangular(n: int) -> int:
    return n * (n + 1) // 2


def prime_factors(n: int) -> list[int]:
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        factors.append(n)
    return factors


def primes_upto(limit: int) -> list[int]:
    if limit < 2:
        return []
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, math.isqrt(limit) + 1):
        if sieve[p]:
            start = p * p
            sieve[start : limit + 1 : p] = b"\x00" * (
                (limit - start) // p + 1
            )
    return [p for p in range(2, limit + 1) if sieve[p]]


def multiplicative_order(base: int, modulo: int, order_factors: list[int]) -> int:
    order = modulo - 1
    residue = base % modulo
    for factor in order_factors:
        while order % factor == 0 and pow(residue, order // factor, modulo) == 1:
            order //= factor
    return order


def minus_one_primes(limit: int, modulo: int, small_primes: list[int]) -> list[int]:
    """Primes q <= limit with q == -1 (mod modulo)."""
    max_k = (limit + 1) // modulo
    if max_k <= 0:
        return []
    sieve = bytearray(b"\x01") * (max_k + 1)
    sieve[0] = 0
    for p in small_primes:
        if p == modulo:
            continue
        residue = pow(modulo % p, -1, p)
        # Start at q >= p^2, otherwise the prime p itself would be marked.
        min_k = (p * p + 1 + modulo - 1) // modulo
        if residue < min_k:
            residue += ((min_k - residue + p - 1) // p) * p
        if residue <= max_k:
            sieve[residue::p] = b"\x00" * ((max_k - residue) // p + 1)
    return [modulo * k - 1 for k in range(1, max_k + 1) if sieve[k]]


def trigger_powers(
    limit: int,
    modulo: int,
    small_primes: list[int],
    order_factors: list[int],
) -> list[tuple[int, int]]:
    events = []
    for q in small_primes:
        if q == modulo:
            continue
        residue = q % modulo
        if residue == 1:
            continue
        order = multiplicative_order(q, modulo, order_factors)
        power = 1
        exponent = 0
        while power <= limit // q:
            power *= q
            exponent += 1
            if (exponent + 1) % order == 0:
                # Exponent 1 for q == -1 (mod p) is handled by the progression
                # sieve, which must include primes much larger than sqrt(limit).
                if not (order == 2 and exponent == 1):
                    events.append((q, power))
    events.sort(key=lambda event: event[1])
    return events


def single_event_sum(limit: int, q: int, q_power: int) -> int:
    m = limit // q_power
    return q_power * (triangular(m) - q * triangular(m // q))


def pair_event_sum(
    limit: int,
    q: int,
    q_power: int,
    r: int,
    r_power: int,
) -> int:
    base = q_power * r_power
    m = limit // base
    return base * (
        triangular(m)
        - q * triangular(m // q)
        - r * triangular(m // r)
        + q * r * triangular(m // (q * r))
    )


def search(limit: int, modulo: int) -> int:
    small_primes = primes_upto(math.isqrt(limit))
    order_factors = prime_factors(modulo - 1)

    linear_events = minus_one_primes(limit, modulo, small_primes)
    higher_events = trigger_powers(limit, modulo, small_primes, order_factors)

    total = 0
    for q in linear_events:
        total += single_event_sum(limit, q, q)
    for q, q_power in higher_events:
        total += single_event_sum(limit, q, q_power)

    for i, q in enumerate(linear_events):
        if q * q > limit:
            break
        end = bisect.bisect_right(linear_events, limit // q)
        for r in linear_events[i + 1 : end]:
            total -= pair_event_sum(limit, q, q, r, r)

    for q, q_power in higher_events:
        end = bisect.bisect_right(linear_events, limit // q_power)
        for r in linear_events[:end]:
            if r != q:
                total -= pair_event_sum(limit, q, q_power, r, r)

    for i, (q, q_power) in enumerate(higher_events):
        if q_power * q_power > limit:
            break
        for r, r_power in higher_events[i + 1 :]:
            if q_power * r_power > limit:
                break
            if q != r:
                total -= pair_event_sum(limit, q, q_power, r, r_power)

    return total


def main() -> None:
    assert search(20, 7) == 49
    assert search(1_000_000, 2017) == 150850429
    assert search(1_000_000_000, 2017) == 249652238344557
    print(search(100_000_000_000, 2017))


if __name__ == "__main__":
    main()
