#!/usr/bin/env python
"""Project Euler 540: Counting Primitive Pythagorean Triples."""

from __future__ import annotations

import math

N = 3141592653589793


def icbrt(n: int) -> int:
    """Floor integer cube root for n >= 0."""
    if n <= 1:
        return n
    r = int(round(n ** (1.0 / 3.0)))
    while (r + 1) ** 3 <= n:
        r += 1
    while r**3 > n:
        r -= 1
    return r


def odd_ge3_count_le(n: int) -> int:
    """Count odd d with 3 <= d <= n."""
    return (n - 1) // 2 if n >= 3 else 0


def raw_opposite_parity_count(limit: int) -> int:
    """Count m > n > 0, m^2+n^2 <= limit, with m and n of opposite parity."""
    if limit < 5:
        return 0

    isqrt = math.isqrt

    # Full rows satisfy m^2 + (m-1)^2 <= limit.
    full = (1 + isqrt(2 * limit - 1)) // 2
    while (full + 1) * (full + 1) + full * full <= limit:
        full += 1
    while full * full + (full - 1) * (full - 1) > limit:
        full -= 1

    k = full // 2
    total = k * k

    m = 2 * k + 1
    if m * m > limit:
        return total

    y = isqrt(limit - m * m)
    while m * m + 1 <= limit:
        rem = limit - m * m
        while y * y > rem:
            y -= 1

        nmax = y if y < m else m - 1
        if m & 1:
            total += nmax // 2
        else:
            total += (nmax + 1) // 2
        m += 1

    return total


def _small_primitive_table(limit: int) -> list[int]:
    """Return P(x) for 0 <= x <= limit using grouped odd-gcd recurrence."""
    isqrt = math.isqrt
    small = [0] * (limit + 1)

    for x in range(1, limit + 1):
        total = raw_opposite_parity_count(x)
        max_d = isqrt(x)
        split = icbrt(x)

        d = 3
        while d <= max_d and d <= split:
            total -= small[x // (d * d)]
            d += 2

        if d <= max_d:
            max_z = x // (d * d)
            for z in range(1, max_z + 1):
                hi = odd_ge3_count_le(isqrt(x // z))
                lo = odd_ge3_count_le(isqrt(x // (z + 1)))
                total -= (hi - lo) * small[z]

        small[x] = total

    return small


def P(limit: int) -> int:
    """Count primitive Pythagorean triples with hypotenuse at most limit."""
    if limit < 5:
        return 0

    isqrt = math.isqrt
    cube = icbrt(limit)
    small = _small_primitive_table(cube)

    # Tail sums:
    #   tail[t] = sum small[limit // s^2] over odd multiples s of t with s > cube.
    # Values below 5 contribute zero, so the useful s range stops at sqrt(limit/5).
    tail = [0] * (cube + 1)
    s_max = isqrt(limit // 5)
    for t in range(1, cube + 1, 2):
        s = (cube // t + 1) * t
        if s % 2 == 0:
            s += t

        acc = 0
        step = 2 * t
        while s <= s_max:
            acc += small[limit // (s * s)]
            s += step
        tail[t] = acc

    transformed = [0] * (cube + 1)
    start = cube if cube & 1 else cube - 1
    for t in range(start, 0, -2):
        x = limit // (t * t)
        total = raw_opposite_parity_count(x)

        s = 3 * t
        step = 2 * t
        while s <= cube:
            total -= transformed[s]
            s += step

        total -= tail[t]
        transformed[t] = total

    return transformed[1]


def main() -> None:
    assert P(20) == 3
    assert P(50) == 7
    assert P(10**6) == 159139
    print(P(N))


if __name__ == "__main__":
    main()
