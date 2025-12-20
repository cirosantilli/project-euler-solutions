#!/usr/bin/env python3
"""
Project Euler 919 - Fortunate Triangles

We call a triangle fortunate if it has integral sides and at least one vertex V
such that distance(V,H) = 1/2 * distance(V,O), where H is orthocentre and O is
circumcentre.

Using AH = 2R|cos(A)| and OA = R, the condition becomes |cos(A)| = 1/4 for at
least one angle.

Define S(P) as the sum of perimeters of all fortunate triangles with perimeter <= P.
We must compute S(10^7).
"""

import math


def _passes_d_filter(m: int, d: int) -> bool:
    """
    We iterate p in the form p = d*m where d ∈ {1,3,5,15}.
    This helper ensures gcd(p, 15) == d.

    - d=1  => m not divisible by 3 or 5
    - d=3  => m not divisible by 5
    - d=5  => m not divisible by 3
    - d=15 => no restriction
    """
    if d == 1:
        return (m % 3) != 0 and (m % 5) != 0
    if d == 3:
        return (m % 5) != 0
    if d == 5:
        return (m % 3) != 0
    return True  # d == 15


def S(P: int) -> int:
    """
    Compute S(P): sum of perimeters a+b+c over all fortunate integer triangles
    with perimeter <= P, with sides ordered a<=b<=c (ordering irrelevant to perimeter).
    """
    total = 0

    def process(raw_a: int, raw_b: int, raw_c: int) -> None:
        """
        Convert the raw parametrised triple into its primitive form by dividing
        by gcd(raw_a, raw_b, raw_c). Then add contribution of all multiples <= P.
        """
        nonlocal total
        g = math.gcd(raw_a, raw_b)
        g = math.gcd(g, raw_c)

        a = raw_a // g
        b = raw_b // g
        c = raw_c // g

        # Avoid double-counting due to swapping the two sides adjacent to
        # the fortunate angle (the equation is symmetric in those two sides).
        if a > b:
            return

        per = a + b + c
        if per > P:
            return

        n = P // per
        total += per * n * (n + 1) // 2

    # Safe q upper bound:
    # For the largest possible gcd scaling (<= 8*15 = 120) and minimal p=1,
    # the reduced perimeter is still Omega(q^2). A safe bound is ~2*sqrt(P).
    q_max = 2 * int(math.isqrt(P)) + 10

    # --- Family A: cos = +1/4 (acute at the fortunate vertex) ---
    # Parametrisation yields raw adjacent sides (A,B) and opposite side C:
    #   A = 8*p*q
    #   B = 15*q^2 - p^2 + 2*p*q
    #   C = 15*q^2 + p^2
    #
    # Constraint for valid triangle in this parametrisation: p < 5q.
    #
    # Perimeter raw form: 10*q*(p+3q)
    # gcd can include odd factors only from gcd(p,15) and a power of two up to 8.
    for q in range(1, q_max + 1):
        qq = q * q
        max_p = 5 * q - 1

        for d in (1, 3, 5, 15):
            # Necessary condition using max possible gcd factor 8*d:
            # 10*q*(p+3q) <= P*(8*d)  -> p <= P*(8*d)/(10q) - 3q
            p_lim = (P * 8 * d) // (10 * q) - 3 * q
            if p_lim > max_p:
                p_lim = max_p
            if p_lim < d:
                continue

            for p in range(d, p_lim + 1, d):
                m = p // d
                if not _passes_d_filter(m, d):
                    continue
                if math.gcd(p, q) != 1:
                    continue

                a = 8 * p * q
                b = 15 * qq - p * p + 2 * p * q
                c = 15 * qq + p * p

                if b <= 0:
                    continue

                process(a, b, c)

    # --- Family B: cos = -1/4 (obtuse at the fortunate vertex) ---
    #   A = 8*p*q
    #   B = 15*q^2 - p^2 - 2*p*q
    #   C = 15*q^2 + p^2
    #
    # Constraint: p < 3q.
    #
    # Perimeter raw form: 6*q*(p+5q)
    for q in range(1, q_max + 1):
        qq = q * q
        max_p = 3 * q - 1

        for d in (1, 3, 5, 15):
            # Necessary condition using max possible gcd factor 8*d:
            # 6*q*(p+5q) <= P*(8*d) -> p <= P*(8*d)/(6q) - 5q
            p_lim = (P * 8 * d) // (6 * q) - 5 * q
            if p_lim > max_p:
                p_lim = max_p
            if p_lim < d:
                continue

            for p in range(d, p_lim + 1, d):
                m = p // d
                if not _passes_d_filter(m, d):
                    continue
                if math.gcd(p, q) != 1:
                    continue

                a = 8 * p * q
                b = 15 * qq - p * p - 2 * p * q
                c = 15 * qq + p * p

                if b <= 0:
                    continue

                process(a, b, c)

    return total


def main() -> None:
    # Test values from the problem statement:
    assert S(10) == 24
    assert S(100) == 3331

    # Required output:
    print(S(10**7))


if __name__ == "__main__":
    main()
