#!/usr/bin/env python
"""
Project Euler 296: Angular Bisector and Tangent

For sides a = BC, b = CA, c = AB with a <= b <= c, the geometry reduces to
BE = a*c/(a+b).  Thus we only need to count ordered side pairs (a,b) and the
multiples of (a+b)/gcd(a,b) that can serve as c.
"""

from math import gcd


LIMIT = 100_000


def brute_force(limit: int) -> int:
    count = 0
    for a in range(1, limit // 3 + 1):
        for b in range(a, (limit - a) // 2 + 1):
            for c in range(b, min(a + b, limit - a - b + 1)):
                if a * c % (a + b) == 0:
                    count += 1
    return count


def count_triangles(limit: int = LIMIT) -> int:
    count = 0
    gcd_ = gcd

    for a in range(1, limit // 3 + 1):
        for b in range(a, (limit - a) // 2 + 1):
            c_max = min(a + b - 1, limit - a - b)
            step = (a + b) // gcd_(a, b)
            count += c_max // step - (b - 1) // step

    return count


def main() -> None:
    assert count_triangles(150) == brute_force(150)
    assert count_triangles(300) == brute_force(300)
    print(count_triangles())


if __name__ == "__main__":
    main()
