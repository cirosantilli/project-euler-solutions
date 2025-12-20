#!/usr/bin/env python3
"""Project Euler 880: Nested Radicals

Compute H(10^15) mod (1031^3 + 2).

A pair (x,y) of non-zero integers is a nested radical pair if:

    sqrt(cuberoot(x) + cuberoot(y)) = cuberoot(a) + cuberoot(b) + cuberoot(c)

for some integers a,b,c, and x/y is not a rational cube.

We must sum (|x|+|y|) over all pairs with |x| <= |y| <= N.

This solver uses a complete parametrisation of primitive (a,b,c) solutions
that force the six cube-root terms in (∛a+∛b+∛c)^2 to collapse to exactly two.
Every solution generates an infinite square-scaling orbit:
    (x,y) -> (k^2 x, k^2 y)
by scaling (a,b,c) -> (k a, k b, k c).

No external libraries are used.
"""

from __future__ import annotations

import math

N = 10**15
MOD = 1031**3 + 2

# Encode signed pairs into a single Python int for fast set membership.
# Values stay within [-N, N]; choose offsets/bases comfortably above that.
OFFSET = 1 << 52  # ~4.5e15
BASE = 1 << 53  # ~9.0e15


# --- Perfect cube check for integers up to about 1e15 ---
# Fast modular filters before an exact integer cube-root refinement.
CUBE_RES_9 = {0, 1, 8}
CUBE_RES_7 = {0, 1, 6}
CUBE_RES_13 = {pow(i, 3, 13) for i in range(13)}


def icbrt(n: int) -> int:
    """Floor integer cube root for n >= 0."""
    if n <= 1:
        return n
    # float seed is safe here (n <= 1e15 in our usage) and corrected below
    x = int(round(n ** (1.0 / 3.0)))
    if x < 0:
        x = 0
    while (x + 1) * (x + 1) * (x + 1) <= n:
        x += 1
    while x * x * x > n:
        x -= 1
    return x


def is_cube_int(z: int) -> bool:
    """True iff z is an integer cube (z may be negative)."""
    if z == 0:
        return True
    n = -z if z < 0 else z
    if n % 9 not in CUBE_RES_9:
        return False
    if n % 7 not in CUBE_RES_7:
        return False
    if n % 13 not in CUBE_RES_13:
        return False
    r = icbrt(n)
    return r * r * r == n


def is_rational_cube_ratio(x: int, y: int) -> bool:
    """True iff x/y is the cube of a rational number."""
    g = math.gcd(abs(x), abs(y))
    xn = x // g
    yn = y // g
    return is_cube_int(xn) and is_cube_int(yn)


def sumsq(k: int) -> int:
    """1^2 + 2^2 + ... + k^2."""
    return k * (k + 1) * (2 * k + 1) // 6


def canon_pair(x: int, y: int) -> tuple[int, int]:
    """Canonical ordering enforcing |x| <= |y|, and deterministic tie-break."""
    ax, ay = abs(x), abs(y)
    if ax > ay or (ax == ay and x > y):
        return y, x
    return x, y


def enc_pair(x: int, y: int) -> int:
    return (x + OFFSET) * BASE + (y + OFFSET)


def iter_primitive_pairs(limit: int):
    """Yield canonical primitive pairs (x,y) with max(|x|,|y|) <= limit.

    Derived from primitive integer triples satisfying one cancellation equation.

    Family A (p odd, gcd(p,q)=1), using q = 2p + s:
        x = 4 q s^3            where s = q - 2p != 0
        y = p (9p + 4s)^3

    Family B (q odd, gcd(p,q)=1), using q = 4p + s with s odd:
        x = 4 p (9p + 2s)^3
        y = q s^3              where s = q - 4p != 0

    These yield all primitive (a,b,c) (up to symmetry), hence all pairs via
    square scaling.
    """
    gcd = math.gcd

    # ---- Family A ----
    # Minimal y for given p occurs at s = 1-2p (i.e., q=1): y_min = p(p+4)^3.
    p = 1
    while True:
        if p * (p + 4) ** 3 > limit:
            break
        # Bound s from y <= limit: 9p + 4s <= cbrt(limit/p)
        t = icbrt(limit // p)
        s_hi = (t - 9 * p) // 4
        s_lo = 1 - 2 * p
        if s_hi >= s_lo:
            for s in range(s_lo, s_hi + 1):
                if s == 0:
                    continue
                if gcd(p, s) != 1:
                    continue
                q = 2 * p + s
                if q <= 0:
                    continue
                x = 4 * q * (s * s * s)
                y = p * (9 * p + 4 * s) ** 3
                ax, ay = abs(x), abs(y)
                m = ay if ay >= ax else ax
                if m > limit:
                    continue
                yield canon_pair(x, y)
        p += 2  # p must be odd for primitiveness

    # ---- Family B ----
    # Minimal x for given p occurs at s = 1-4p (i.e., q=1): x_min = 4p(p+2)^3.
    p = 1
    while True:
        if 4 * p * (p + 2) ** 3 > limit:
            break
        # Bound s from x <= limit: 9p + 2s <= cbrt(limit/(4p))
        t = icbrt(limit // (4 * p))
        s_hi = (t - 9 * p) // 2
        s_lo = 1 - 4 * p
        # s must be odd so that q=4p+s is odd (primitive constraint)
        if s_lo % 2 == 0:
            s_lo += 1
        if s_hi >= s_lo:
            for s in range(s_lo, s_hi + 1, 2):
                if s == 0:
                    continue
                if gcd(p, s) != 1:
                    continue
                q = 4 * p + s
                if q <= 0:
                    continue
                x = 4 * p * (9 * p + 2 * s) ** 3
                y = q * (s * s * s)
                ax, ay = abs(x), abs(y)
                m = ay if ay >= ax else ax
                if m > limit:
                    continue
                yield canon_pair(x, y)
        p += 1  # no parity restriction on p in this family


def H_mod(limit: int) -> int:
    """Compute H(limit) modulo MOD."""
    seen: set[int] = set()
    total = 0
    isqrt = math.isqrt

    for x, y in iter_primitive_pairs(limit):
        # Filter out rational cube ratios
        if is_rational_cube_ratio(x, y):
            continue
        key = enc_pair(x, y)
        if key in seen:
            continue
        seen.add(key)

        ax, ay = abs(x), abs(y)
        # (x,y) is canonical so ay >= ax
        kmax = isqrt(limit // ay)
        if kmax <= 0:
            continue
        contrib = ((ax + ay) % MOD) * (sumsq(kmax) % MOD)
        total = (total + contrib) % MOD

    return total


def _self_test() -> None:
    # Example pairs given in the problem statement.
    # Ensure our primitive generator contains them at an adequate bound.
    check_limit = 6000
    pairs = set(iter_primitive_pairs(check_limit))
    assert (-4, 125) in pairs
    assert (5, 5324) in pairs

    # Given statement value.
    assert H_mod(10**3) == 2535


def main() -> None:
    _self_test()
    print(H_mod(N) % MOD)


if __name__ == "__main__":
    main()
