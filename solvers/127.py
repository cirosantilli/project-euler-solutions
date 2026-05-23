#!/usr/bin/env python
from math import gcd


def solve(limit: int) -> int:
    rad = [1] * limit
    for i in range(2, limit):
        if rad[i] > 1:
            continue
        for j in range(i, limit, i):
            rad[j] *= i

    candidates = sorted((rad[n], n) for n in range(1, limit))

    total = 0
    for c in range(3, limit):
        rad_c = rad[c]
        cutoff = c // rad_c
        for rad_a, a in candidates:
            if rad_a >= cutoff:
                break
            if 2 * a >= c:
                continue
            b = c - a
            if rad_a * rad[b] * rad_c < c and gcd(rad_a, rad[b]) == 1:
                total += c
    return total


def main() -> None:
    assert solve(1000) == 12523
    print(solve(120000))


if __name__ == "__main__":
    main()
