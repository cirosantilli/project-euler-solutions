#!/usr/bin/env python3
"""
Project Euler 804: Counting Binary Quadratic Representations

We count integer pairs (x, y) with:
    x^2 + x*y + 41*y^2 <= N
and subtract the origin so that only positive n are counted.
"""

from __future__ import annotations

import math


def g(n: int) -> int:
    """Brute-force g(n) for small n (used only for statement asserts)."""
    cnt = 0
    # 41*y^2 <= n  => |y| <= sqrt(n/41)
    y_lim = math.isqrt(n // 41) + 2
    for y in range(-y_lim, y_lim + 1):
        # Solve x^2 + y*x + (41*y^2 - n) = 0
        D = 4 * n - 163 * y * y
        if D < 0:
            continue
        s = math.isqrt(D)
        if s * s != D:
            continue
        # Roots: (-y ± s)/2, must be integer
        if (-y + s) & 1 == 0:
            cnt += 1
        if s != 0 and ((-y - s) & 1 == 0):
            cnt += 1
    return cnt


def T(N: int) -> int:
    """
    Compute T(N) = sum_{n=1..N} g(n).

    Using:
        x^2 + x*y + 41*y^2 = ((2x + y)^2 + 163*y^2) / 4
    Let u = 2x + y. Then u ≡ y (mod 2) and:
        u^2 + 163*y^2 <= 4N

    For each y, let t = floor_sqrt(4N - 163*y^2).
    The number of integers u with |u| <= t and u ≡ y (mod 2) is:
      - y even:  2*floor(t/2) + 1  == t + 1 - (t&1)
      - y odd:   2*floor((t+1)/2) == t + (t&1)

    Sum over y with symmetry and subtract the origin.
    """
    if N <= 0:
        return 0

    A = 4 * N
    B = 163

    ymax = math.isqrt(A // B)

    # y = 0 case
    t0 = math.isqrt(A)
    total = t0 + 1 - (t0 & 1)  # even u

    # Incrementally maintain B*y^2 without multiplications:
    # y^2 increases by odd numbers: 1,3,5,...; multiplying by B gives increments by B,3B,5B,...
    By2 = 0
    Bdelta = B
    twoB = 2 * B

    for y in range(1, ymax + 1):
        By2 += Bdelta
        Bdelta += twoB
        t = math.isqrt(A - By2)

        if y & 1:
            c = t + (t & 1)  # odd u
        else:
            c = t + 1 - (t & 1)  # even u

        total += 2 * c  # y and -y contribute equally

    return total - 1  # remove (u,y)=(0,0) => (x,y)=(0,0) which corresponds to n=0


def main() -> None:
    # Asserts from the problem statement
    assert g(53) == 4
    assert T(10**3) == 474
    assert T(10**6) == 492_128

    print(T(10**16))


if __name__ == "__main__":
    main()
