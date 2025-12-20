#!/usr/bin/env python3
"""
Project Euler 876: Triplet Tricks

We start from (a,b,c) and repeatedly apply one of three involutions:
    a <- 2(b+c) - a
    b <- 2(c+a) - b
    c <- 2(a+b) - c

Let f(a,b,c) be the minimum number of steps until some entry becomes 0
(or 0 if it's impossible). Define F(a,b) = sum_{c>=1} f(a,b,c).

Key facts used by this program (explained in README.md):
  1) f(a,b,c) > 0  <=>  (c-a-b)^2 - 4ab is a perfect square.
  2) All such c can be generated from divisors x|a and y|b via:
         c = (x + y)(a/x + b/y)
         c = (x - y)(a/x - b/y)  (keep only if positive)
  3) The minimal step count equals the number of subtraction steps in the
     Euclidean algorithm for (x,y), and is one less for the second formula.

We enumerate all divisor pairs (x,y), generate candidate c values, keep the
minimum step count per c, and sum them.
"""


def euclid_subtraction_steps(m: int, n: int) -> int:
    """
    Number of steps of the subtractive Euclidean algorithm to reduce (m,n)
    to (g,0), i.e. repeatedly replace the larger by (larger - smaller).

    This equals the sum of quotients in the standard Euclidean algorithm.
    """
    steps = 0
    while n:
        q = m // n
        steps += q
        m, n = n, m - q * n
    return steps


def divisors_6pow(k: int):
    """All positive divisors of 6^k = 2^k * 3^k."""
    p2 = [1]
    p3 = [1]
    for _ in range(k):
        p2.append(p2[-1] * 2)
        p3.append(p3[-1] * 3)
    return [p2[i] * p3[j] for i in range(k + 1) for j in range(k + 1)]


def divisors_10pow(k: int):
    """All positive divisors of 10^k = 2^k * 5^k."""
    p2 = [1]
    p5 = [1]
    for _ in range(k):
        p2.append(p2[-1] * 2)
        p5.append(p5[-1] * 5)
    return [p2[i] * p5[j] for i in range(k + 1) for j in range(k + 1)]


def compute_F_for_powers(k: int) -> int:
    """
    Compute F(6^k, 10^k).
    """
    # 6^k = 2^k * 3^k, 10^k = 2^k * 5^k
    a = pow(6, k)
    b = pow(10, k)

    div_a = divisors_6pow(k)
    div_b = divisors_10pow(k)

    # Precompute complementary factors to avoid repeated divisions.
    xa = [(x, a // x) for x in div_a]
    yb = [(y, b // y) for y in div_b]

    best = {}  # c -> minimal f(a,b,c)

    for x, u in xa:
        for y, v in yb:
            s = euclid_subtraction_steps(x, y)

            # First family: (x+y)(u+v)
            c1 = (x + y) * (u + v)
            prev = best.get(c1)
            if prev is None or s < prev:
                best[c1] = s

            # Second family: (x-y)(u-v)
            c2 = (x - y) * (u - v)
            if c2 > 0:
                s2 = s - 1
                if s2 > 0:
                    prev = best.get(c2)
                    if prev is None or s2 < prev:
                        best[c2] = s2

    return sum(best.values())


def solve(limit_k: int = 18) -> int:
    return sum(compute_F_for_powers(k) for k in range(1, limit_k + 1))


def _self_test():
    # Given examples from the problem statement.
    assert compute_F_for_powers(1) == 17
    assert compute_F_for_powers(2) == 179


if __name__ == "__main__":
    _self_test()
    print(solve(18))
