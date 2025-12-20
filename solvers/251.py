#!/usr/bin/env python3
"""
Project Euler 251: Cardano Triplets

A triplet (a,b,c) of positive integers is a Cardano triplet if:

    cbrt(a + b*sqrt(c)) + cbrt(a - b*sqrt(c)) = 1

This program counts how many such triplets satisfy a + b + c <= LIMIT.
It avoids floating point root computations by using an equivalent integer formulation.
"""

from __future__ import annotations

import math
import sys


def is_cardano_triplet(a: int, b: int, c: int) -> bool:
    """Check the defining condition using an equivalent integer identity.

    From algebraic manipulation of the defining equation one can show:
        27*b^2*c = (a+1)^2 * (8a-1)
    for all integer Cardano triplets (a,b,c).
    """
    if a <= 0 or b <= 0 or c <= 0:
        return False
    return 27 * b * b * c == (a + 1) * (a + 1) * (8 * a - 1)


def count_cardano_triplets(limit_sum: int) -> int:
    """
    Count Cardano triplets (a,b,c) with a+b+c <= limit_sum.

    Key parametrisation (all variables positive integers):

        a = 3*p*q - 1
        b = p*r
        c = s*q^2
        8*p*q = s*r^2 + 3
        gcd(q, r) = 1
        r is odd

    For fixed (q, r) with gcd(q,r)=1 and r odd, the solutions (p,s) form an
    arithmetic progression; we count how many satisfy the sum constraint without
    enumerating each triplet explicitly.
    """
    if limit_sum < 0:
        return 0

    # We rewrite a+b+c <= limit_sum as:
    #     p(3q+r) + s q^2 <= limit_sum + 1
    L = limit_sum + 1

    # Since c = s*q^2 >= q^2, we must have q^2 <= limit_sum.
    q_max = math.isqrt(limit_sum)
    total = 0

    for q in range(1, q_max + 1):
        q2 = q * q

        # --- r = 1 special case ---
        # When r=1, the divisibility condition is automatic (r^2=1).
        # Equation 8*p*q = s + 3 gives s = 8*p*q - 3.
        # Inequality becomes:
        #   p*(8*q^3 + 3*q + 1) <= (limit_sum + 1) + 3*q^2
        den = 8 * q * q2 + 3 * q + 1
        p_max = (L + 3 * q2) // den
        total += p_max  # counts p = 1..p_max

        # --- r >= 3, odd ---
        # A simple (safe) upper bound: p*r >= (r^2+3)/(8q) * r > r^3/(8q),
        # so if r^3 > 8*q*L then no solution can fit within the limit.
        # (We slightly overshoot to stay safe.)
        r_max = int((8 * q * L) ** (1.0 / 3.0)) + 3
        if r_max % 2 == 0:
            r_max += 1

        for r in range(3, r_max + 1, 2):
            if math.gcd(q, r) != 1:
                continue

            r2 = r * r

            # Find the smallest positive p such that:
            #     8*p*q ≡ 3 (mod r^2)
            # Since gcd(8q, r^2)=1 (r odd and gcd(q,r)=1), an inverse exists.
            inv = pow((8 * q) % r2, -1, r2)
            p0 = (3 * inv) % r2
            if p0 == 0:
                p0 = r2

            # Corresponding s0 from 8*p0*q = s0*r^2 + 3
            s0 = (8 * p0 * q - 3) // r2

            base = p0 * (3 * q + r) + s0 * q2
            if base > L:
                continue

            step = r2 * (3 * q + r) + 8 * q * q2  # increase when p->p+r^2

            total += (L - base) // step + 1

    return total


def main() -> None:
    # Problem statement test values:
    assert is_cardano_triplet(2, 1, 5)
    assert count_cardano_triplets(1000) == 149

    limit = 110_000_000
    if len(sys.argv) >= 2:
        limit = int(sys.argv[1])

    print(count_cardano_triplets(limit))


if __name__ == "__main__":
    main()
