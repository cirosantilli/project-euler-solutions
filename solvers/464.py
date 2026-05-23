#!/usr/bin/env python
"""
Project Euler 464: Möbius function and intervals

Goal: compute C(20_000_000) and print it.

No external libraries are used (only Python standard library).
"""

from __future__ import annotations

from array import array


def mobius_sieve(n: int) -> array:
    """Return an array('b') mu[0..n] containing Möbius values (-1, 0, 1).

    Uses a linear sieve (Euler sieve), O(n).
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    mu = array("b", [0]) * (n + 1)
    if n == 0:
        return mu

    lp = array("I", [0]) * (n + 1)  # least prime factor
    primes: list[int] = []

    mu[1] = 1
    append_prime = primes.append

    for i in range(2, n + 1):
        if lp[i] == 0:
            lp[i] = i
            append_prime(i)
            mu[i] = -1
        li = lp[i]
        mi = mu[i]
        for p in primes:
            if p > li:
                break
            ip = i * p
            if ip > n:
                break
            lp[ip] = p
            if p == li:
                mu[ip] = 0
                break
            mu[ip] = -mi

    return mu


def compute_C(n: int) -> int:
    """Compute C(n) as defined in the problem statement."""
    if n <= 0:
        return 0

    mu = mobius_sieve(n)
    prefix_a = prefix_b = 0
    min_a = max_a = 0
    min_b = max_b = 0

    for pos in range(1, n + 1):
        m = mu[pos]
        if m == 1:
            prefix_a += 99
            prefix_b -= 100
        elif m == -1:
            prefix_a -= 100
            prefix_b += 99

        if prefix_a < min_a:
            min_a = prefix_a
        elif prefix_a > max_a:
            max_a = prefix_a

        if prefix_b < min_b:
            min_b = prefix_b
        elif prefix_b > max_b:
            max_b = prefix_b

    range_a = max_a - min_a + 1
    range_b = max_b - min_b + 1
    bit_a = array("I", [0]) * (range_a + 2)
    bit_b = array("I", [0]) * (range_b + 2)
    offset_a = 1 - min_a
    offset_b = 1 - min_b
    limit_a = range_a + 1
    limit_b = range_b + 1

    pos = offset_a
    while pos <= limit_a:
        bit_a[pos] += 1
        pos += pos & -pos

    pos = offset_b
    while pos <= limit_b:
        bit_b[pos] += 1
        pos += pos & -pos

    prefix_a = prefix_b = 0
    bad_a = bad_b = 0

    for pos in range(1, n + 1):
        m = mu[pos]
        if m == 1:
            prefix_a += 99
            prefix_b -= 100
        elif m == -1:
            prefix_a -= 100
            prefix_b += 99

        idx = prefix_a + offset_a
        subtotal = 0
        t = idx - 1
        while t:
            subtotal += bit_a[t]
            t -= t & -t
        bad_a += subtotal

        t = idx
        while t <= limit_a:
            bit_a[t] += 1
            t += t & -t

        idx = prefix_b + offset_b
        subtotal = 0
        t = idx - 1
        while t:
            subtotal += bit_b[t]
            t -= t & -t
        bad_b += subtotal

        t = idx
        while t <= limit_b:
            bit_b[t] += 1
            t += t & -t

    return n * (n + 1) // 2 - bad_a - bad_b


def main() -> None:
    # Statement checks
    mu10 = mobius_sieve(10)
    assert sum(1 for k in range(2, 11) if mu10[k] == 1) == 2  # P(2,10)
    assert sum(1 for k in range(2, 11) if mu10[k] == -1) == 4  # N(2,10)

    assert compute_C(10) == 13
    assert compute_C(500) == 16676
    assert compute_C(10_000) == 20155319

    print(compute_C(20_000_000))


if __name__ == "__main__":
    main()
