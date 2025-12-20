#!/usr/bin/env python3
"""
Project Euler 748: Upside Down Diophantine Equation

We need S(N) = sum(x+y+z) over primitive integer solutions (x,y,z) with:
    1/x^2 + 1/y^2 = 13/z^2
    1 <= x,y,z <= N
    x <= y
    gcd(x,y,z) = 1

This program computes the last 9 digits of S(10^16).
No external libraries are used.
"""

from __future__ import annotations

import math


def fourth_root_floor(n: int) -> int:
    """Return floor(n^(1/4)) for n >= 0, using integer arithmetic."""
    if n <= 0:
        return 0
    x = math.isqrt(math.isqrt(n))
    while (x + 1) ** 4 <= n:
        x += 1
    while x**4 > n:
        x -= 1
    return x


def S(N: int, mod: int | None = None) -> int:
    """
    Compute S(N).
    If mod is not None, return S(N) % mod efficiently.
    """
    # Safe global bound:
    # If p >= q and p^2 + q^2 = 13 r^2 then p^2 >= (13/2) r^2, so
    # y = p*r implies y^2 >= (13/2) r^4. If y <= N then r^4 <= 2N^2/13.
    r_max = fourth_root_floor((2 * N * N) // 13)

    m_max = math.isqrt(r_max)  # since r = m^2 + n^2 <= r_max
    sq = [i * i for i in range(m_max + 1)]

    gcd = math.gcd
    isqrt = math.isqrt

    total = 0
    THRESH = 10**20  # periodic reduction threshold when mod is used

    for m in range(1, m_max + 1):
        mm = sq[m]
        n_max = isqrt(r_max - mm)

        # Opposite parity: (m - n) odd <=> m and n have different parity.
        n_start = 0 if (m & 1) else 1

        for n in range(n_start, n_max + 1, 2):
            if gcd(m, n) != 1:
                continue

            nn = sq[n]
            r = mm + nn

            # Gaussian integers:
            # (p + i q) = (3 + 2i) * (m + i n)^2
            u = mm - nn  # m^2 - n^2
            v = 2 * m * n  # 2mn

            # Multiply (3 + 2i)(u + i v):
            # real = 3u - 2v, imag = 3v + 2u
            a = abs(3 * u - 2 * v)
            b = abs(3 * v + 2 * u)  # abs is essential: imag can be negative

            # Order so that p >= q (which gives x <= y after mapping back).
            if a < b:
                p, q = b, a
            else:
                p, q = a, b

            # Non-primitive family occurs exactly when 13 | p and 13 | q (then 13 | r).
            if (p % 13 == 0) and (q % 13 == 0):
                continue

            # Convert back to the original (x,y,z):
            x = q * r
            y = p * r
            if x > N or y > N:
                continue
            z = p * q
            if z > N:
                continue

            s = x + y + z
            if mod is None:
                total += s
            else:
                total += s
                if total >= THRESH:
                    total %= mod

    return total if mod is None else (total % mod)


def main() -> None:
    # Checks given in the problem statement
    assert S(10**2) == 124
    assert S(10**3) == 1470
    assert S(10**5) == 2340084

    ans = S(10**16, mod=10**9)
    print(f"{ans:09d}")


if __name__ == "__main__":
    main()
