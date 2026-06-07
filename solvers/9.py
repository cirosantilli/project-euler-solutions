#!/usr/bin/env python
from __future__ import annotations

def solve(total: int) -> int:
    best = 0
    m = 2
    while 2 * m * (m + 1) <= total:
        for n in range(1, m):
            s = 2 * m * (m + n)
            if total % s != 0:
                continue
            k = total // s
            a = k * (m * m - n * n)
            b = k * (2 * m * n)
            c = k * (m * m + n * n)
            a, b, c = sorted((a, b, c))
            if a > 0 and a * a + b * b == c * c and a + b + c == total:
                best = max(best, a * b * c)
        m += 1
    if best:
        return best
    raise ValueError(f"No Pythagorean triplet sums to {total}")

if __name__ == "__main__":
    assert solve(12) == 60
    assert solve(210) == 328860
    print(solve(1000))
