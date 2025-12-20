#!/usr/bin/env python3
"""
Project Euler 785: Symmetric Diophantine Equation

We need S(N) = sum_{(x,y,z)} (x+y+z)
over primitive solutions of:
    15(x^2+y^2+z^2) = 34(xy+yz+zx)
with 1 <= x <= y <= z <= N and gcd(x,y,z)=1.

This implementation uses a parameterisation into coprime pairs (a,b).
"""

import math


def solve(N: int) -> int:
    """
    Compute S(N).
    """

    # Upper bound on 'a' derived from:
    # One coordinate is B = 5a^2 - 2ab >= 3a^2 (since b < a),
    # so if max(x,y,z) <= N then 3a^2 <= N, thus a <= floor(sqrt(N/3)).
    max_a = math.isqrt(N // 3)

    # Smallest prime factor sieve for fast coprimality checks
    spf = list(range(max_a + 1))
    for i in range(2, int(math.isqrt(max_a)) + 1):
        if spf[i] == i:  # prime
            step = i
            start = i * i
            for j in range(start, max_a + 1, step):
                if spf[j] == j:
                    spf[j] = i

    def distinct_prime_factors(x: int):
        """Return list of distinct prime factors of x."""
        res = []
        while x > 1:
            p = spf[x]
            res.append(p)
            while x % p == 0:
                x //= p
        return res

    total = 0

    # Precompute constants involving N used in square-roots
    # (done once to reduce attribute lookups)
    N3 = 12 * N
    N5 = 5 * N

    for a in range(1, max_a + 1):
        aa = a * a
        pf = distinct_prime_factors(a)

        # b >= 1
        # B = 5a^2 - 2ab <= N  ==>  b >= (5a^2 - N)/(2a)
        if 5 * aa <= N:
            bmin = 1
        else:
            num = 5 * aa - N
            bmin = (num + (2 * a - 1)) // (2 * a)
            if bmin < 1:
                bmin = 1

        # C = 3a^2 - 8ab + 5b^2 must be > 0.
        # This quadratic becomes negative for b in (3a/5, a), so we require b < 3a/5.
        bmax_pos = (3 * a - 1) // 5
        if bmax_pos < bmin:
            continue

        # A = 2ab + 3b^2 <= N -> 3b^2 + 2ab - N <= 0
        # Solve for b: b <= floor((-2a + sqrt(4a^2 + 12N))/6)
        disc1 = 4 * aa + N3
        bmax1 = (-2 * a + math.isqrt(disc1)) // 6

        # C = 3a^2 - 8ab + 5b^2 <= N
        # 5b^2 - 8ab + (3a^2 - N) <= 0
        # roots: (4a ± sqrt(a^2 + 5N))/5
        disc3 = aa + N5
        sdisc3 = math.isqrt(disc3)
        bmax3 = (4 * a + sdisc3) // 5
        bmin3 = (4 * a - sdisc3 + 4) // 5  # ceil

        if bmin3 > bmin:
            bmin = bmin3

        bmax = bmax_pos
        if bmax1 < bmax:
            bmax = bmax1
        if bmax3 < bmax:
            bmax = bmax3
        if bmax >= a:
            bmax = a - 1

        if bmax < bmin:
            continue

        # Iterate possible b
        for b in range(bmin, bmax + 1):

            # Fast coprimality: check b not divisible by any prime factor of a
            ok = True
            for p in pf:
                if b % p == 0:
                    ok = False
                    break
            if not ok:
                continue

            ab = a * b
            bb = b * b

            # The three coordinates (in some order)
            A = 2 * ab + 3 * bb
            B = 5 * aa - 2 * ab
            C = 3 * aa - 8 * ab + 5 * bb

            # Must be positive
            if C <= 0:
                continue

            # Bound check
            if A > N or B > N or C > N:
                continue

            # Primitive condition: gcd(x,y,z) can only be 1 or 19 here.
            # Non-primitive exactly when all are divisible by 19.
            if A % 19 == 0 and B % 19 == 0:
                continue

            # Contribution is x+y+z = 8 * (a^2 - ab + b^2)
            total += 8 * (aa - ab + bb)

    return total


def _test():
    # Test value given in the problem statement:
    # For N=10^2, solutions are (1,7,16), (8,9,39), (11,21,72) so S(100)=184.
    assert solve(10**2) == 184


def main():
    _test()
    print(solve(10**9))


if __name__ == "__main__":
    main()
