#!/usr/bin/env python
from __future__ import annotations

import math


def max_index(limit: int) -> int:
    """Largest n with 2*n^2 + 2*n + 1 < limit."""
    if limit <= 5:
        return 0

    n = (math.isqrt(2 * limit - 1) - 1) // 2
    while 2 * n * n + 2 * n + 1 >= limit:
        n -= 1
    while 2 * (n + 1) * (n + 1) + 2 * (n + 1) + 1 < limit:
        n += 1
    return n


def solve(limit: int) -> int:
    last = max_index(limit)
    if last == 0:
        return 0

    residuals = [0] * (last + 1)
    for n in range(1, last + 1):
        residuals[n] = 2 * n * n + 2 * n + 1

    composite = bytearray(last + 1)
    count = 0

    for i in range(1, last + 1):
        if composite[i] == 0:
            count += 1

        p = residuals[i]
        if p <= 1 or p > i + last + 1:
            continue

        # Q_n - Q_i = 2(n-i)(n+i+1), so a factor p of Q_i hits the
        # future indices n == i (mod p) and n == -i-1 (mod p).
        start = i + p
        if start <= last:
            for j in range(start, last + 1, p):
                composite[j] = 1
                value = residuals[j]
                while value % p == 0:
                    value //= p
                residuals[j] = value

        start = (-i - 1) % p
        if start == 0:
            start = p
        if start <= i:
            start += ((i - start) // p + 1) * p
        if start <= last:
            for j in range(start, last + 1, p):
                composite[j] = 1
                value = residuals[j]
                while value % p == 0:
                    value //= p
                residuals[j] = value

    return count


def main() -> None:
    assert solve(1000) == 10
    assert solve(10**6) == 175
    print(solve(5_000_000_000_000_000))


if __name__ == "__main__":
    main()
