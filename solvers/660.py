#!/usr/bin/env python3
"""
Project Euler 660
Pandigital triangles with a 120° angle.

We look for integer-sided triangles where one angle is 120°. If the side opposite
that angle is L and the other sides are x,y, then:

    L^2 = x^2 + y^2 + x*y

A triangle is n-pandigital if, when its three side lengths are written in base n
(without leading zeros), the digits 0..n-1 appear exactly once in total.

This program prints the required sum for 9 <= n <= 18.

No external libraries are used, and everything runs on a single CPU core.
"""

from math import gcd, isqrt


# ----------------------------
# Small utilities
# ----------------------------


def egcd(a: int, b: int):
    """Extended gcd: returns (g,x,y) with ax+by=g."""
    x0, y0, x1, y1 = 1, 0, 0, 1
    while b:
        q = a // b
        a, b = b, a - q * b
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    return a, x0, y0


def modinv(a: int, m: int) -> int:
    """Modular inverse of a mod m (assumes gcd(a,m)=1)."""
    g, x, _ = egcd(a, m)
    # In this program we only call modinv when the inverse must exist.
    assert g == 1
    return x % m


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# ----------------------------
# Base-digit helpers (used in asserts only)
# ----------------------------


def to_base_digits(x: int, base: int):
    """Return digits of x in given base, most-significant first."""
    assert x > 0 and base >= 2
    ds = []
    while x:
        x, r = divmod(x, base)
        ds.append(r)
    ds.reverse()
    return ds


def is_pandigital_triangle(a: int, b: int, c: int, base: int) -> bool:
    """Slow, clear checker: used for the problem-statement example assert."""
    L = max(a, b, c)
    if L == a:
        x, y = b, c
    elif L == b:
        x, y = a, c
    else:
        x, y = a, b
    if L * L != x * x + y * y + x * y:
        return False

    seen = 0
    cnt = 0
    full = (1 << base) - 1
    for v in (a, b, c):
        for d in to_base_digits(v, base):
            bit = 1 << d
            if seen & bit:
                return False
            seen |= bit
            cnt += 1
    return cnt == base and seen == full


# Problem statement example: (217, 248, 403) is 9-pandigital.
assert 403 * 403 == 217 * 217 + 248 * 248 + 217 * 248
assert is_pandigital_triangle(217, 248, 403, 9)


# ----------------------------
# Fast digit-mask tables
# ----------------------------


def precompute_tables(base: int):
    """
    Precompute digit-bitmasks for fixed-length chunks:

    t1[d] = mask for 1-digit value d
    t2[v] = mask for exact 2-digit value v (leading digit nonzero) or -1 if invalid/repeated
    t3[v] = mask for exact 3-digit value v (leading digit nonzero) or -1 if invalid/repeated
    t3z[v] = mask for 3-digit chunk v where leading zeros ARE allowed (used for low chunks)

    Also returns b2=base^2, b3=base^3.
    """
    t1 = [1 << d for d in range(base)]

    b2 = base * base
    b3 = b2 * base

    t2 = [-1] * (base * base)
    for v in range(base, base * base):  # exact 2-digit numbers start at '10'
        d0 = v // base
        d1 = v - d0 * base
        if d0 != d1:
            t2[v] = (1 << d0) | (1 << d1)

    t3 = [-1] * b3
    for v in range(b2, b3):  # exact 3-digit numbers start at '100'
        d0 = v // b2
        r = v - d0 * b2
        d1 = r // base
        d2 = r - d1 * base
        if d0 != d1 and d0 != d2 and d1 != d2:
            t3[v] = (1 << d0) | (1 << d1) | (1 << d2)

    t3z = [-1] * b3
    for v in range(b3):  # allow leading zeros
        d0 = v // b2
        r = v - d0 * b2
        d1 = r // base
        d2 = r - d1 * base
        if d0 != d1 and d0 != d2 and d1 != d2:
            t3z[v] = (1 << d0) | (1 << d1) | (1 << d2)

    return t1, t2, t3, t3z, b2, b3


def mask_fixed_len(x: int, d: int, base: int, tables) -> int:
    """
    Return the digit mask of x assuming x has EXACTLY d base-'base' digits.
    Returns -1 if any digit repeats within x.
    """
    t1, t2, t3, t3z, _b2, b3 = tables
    if d == 1:
        return t1[x]
    if d == 2:
        return t2[x]  # x must be in [base, base^2-1]
    if d == 3:
        return t3[x]  # x must be in [base^2, base^3-1]

    hi, lo = divmod(x, b3)
    if d == 4:
        mh = t1[hi]
        ml = t3z[lo]
    elif d == 5:
        mh = t2[hi]
        ml = t3z[lo]
    else:  # d == 6
        mh = t3[hi]
        ml = t3z[lo]

    if mh == -1 or ml == -1 or (mh & ml):
        return -1
    return mh | ml


# ----------------------------
# Main solver
# ----------------------------


def solve() -> int:
    # Precompute per-base settings for 9..18.
    limit = [0] * 19
    powb = [None] * 19
    combos = [None] * 19
    full_mask = [0] * 19
    modM = [0] * 19
    target = [0] * 19
    tables = [None] * 19

    for base in range(9, 19):
        D = (base + 2) // 3  # max digit count for the largest side in base 'base'
        # From triangle inequality: if L has D digits then other sides have >= D-1 digits.
        # Total digits is 'base', so 3D-2 <= base => D <= floor((base+2)/3).
        limit[base] = base**D - 1

        # powb[i] = base^i, i up to D+1
        p = [1]
        for _ in range(D + 1):
            p.append(p[-1] * base)
        powb[base] = p

        cs = []
        for da in range(1, D + 1):
            for db in range(1, D + 1):
                dc = base - da - db
                if 1 <= dc <= D:
                    cs.append((da, db, dc))
        combos[base] = cs

        full_mask[base] = (1 << base) - 1
        modM[base] = base - 1
        target[base] = 0 if (base % 2 == 0) else (base - 1) // 2
        tables[base] = precompute_tables(base)

    # Special fast path data for base 18 (it dominates the workload).
    t1_18, t2_18, t3_18, t3z_18, b2_18, b3_18 = tables[18]
    full18 = full_mask[18]
    pow18 = powb[18]
    M18 = modM[18]  # 17

    sums = [0] * 19

    max_a = limit[18]
    max_m = isqrt(max_a) + 2

    for m in range(2, max_m):
        disc = 4 * max_a - 3 * m * m
        if disc <= 0:
            continue
        nmax = (-m + isqrt(disc)) // 2
        if nmax >= m:
            nmax = m - 1

        mm = m * m
        for n in range(1, nmax + 1):
            if (m - n) % 3 == 0:
                continue
            if gcd(m, n) != 1:
                continue

            # Primitive 120° triangle with largest side 'a'.
            a = mm + m * n + n * n
            if a > max_a:
                break
            b = 2 * m * n + n * n
            c = mm - n * n

            # Start base shortcut using increasing limits.
            if a <= limit[15]:
                start_base = 9
            elif a <= limit[16]:
                start_base = 16
            elif a <= limit[17]:
                start_base = 17
            else:
                start_base = 18

            # --------------------
            # Base 18 (fixed 6+6+6 digits, and modulus is prime)
            # --------------------
            s18 = (a + b + c) % M18
            step18 = 1 if s18 == 0 else 17

            # All three sides must be 6 digits in base 18:
            # 18^5 <= k*side <= 18^6 - 1
            L = ceil_div(pow18[5], a)
            R = (pow18[6] - 1) // a
            L2 = ceil_div(pow18[5], b)
            R2 = (pow18[6] - 1) // b
            if L2 > L:
                L = L2
            if R2 < R:
                R = R2
            if L <= R:
                L3 = ceil_div(pow18[5], c)
                R3 = (pow18[6] - 1) // c
                if L3 > L:
                    L = L3
                if R3 < R:
                    R = R3
                if L <= R:
                    if L < 1:
                        L = 1
                    if step18 == 1:
                        k = L
                        A = a * k
                        B = b * k
                        C = c * k
                        incA, incB, incC = a, b, c
                    else:
                        # smallest multiple of 17 >= L
                        k = ((L + 16) // 17) * 17
                        A = a * k
                        B = b * k
                        C = c * k
                        incA, incB, incC = a * 17, b * 17, c * 17

                    while k <= R:
                        # mask for 6 base-18 digits via split into 3+3 digits
                        hi, lo = divmod(A, b3_18)
                        ma = t3_18[hi]
                        if ma != -1:
                            ml = t3z_18[lo]
                            if ml != -1 and (ma & ml) == 0:
                                ma |= ml

                                hi, lo = divmod(B, b3_18)
                                mb = t3_18[hi]
                                if mb != -1:
                                    ml = t3z_18[lo]
                                    if ml != -1 and (mb & ml) == 0:
                                        mb |= ml

                                        if (ma & mb) == 0:
                                            hi, lo = divmod(C, b3_18)
                                            mc = t3_18[hi]
                                            if mc != -1:
                                                ml = t3z_18[lo]
                                                if ml != -1 and (mc & ml) == 0:
                                                    mc |= ml
                                                    if (
                                                        (ma & mc) == 0
                                                        and (mb & mc) == 0
                                                        and (ma | mb | mc) == full18
                                                    ):
                                                        sums[18] += A

                        k += step18
                        A += incA
                        B += incB
                        C += incC

            # --------------------
            # Other bases (9..17) when relevant for this primitive triple
            # --------------------
            for base in range(start_base, 18):
                if a > limit[base]:
                    continue

                p = powb[base]
                cs = combos[base]
                t = tables[base]
                fm = full_mask[base]
                M = modM[base]
                targ = target[base]

                s = (a + b + c) % M

                if base % 2 == 0:
                    # target is 0 -> k*s ≡ 0 (mod M)
                    if s == 0:
                        step = 1
                        residue = 0
                    else:
                        g = gcd(s, M)
                        step = M // g
                        residue = 0
                else:
                    # target is (M/2) -> solve k*s ≡ targ (mod M)
                    if s == 0:
                        if targ != 0:
                            continue
                        step = 1
                        residue = 0
                    else:
                        g = gcd(s, M)
                        if targ % g:
                            continue
                        step = M // g
                        inv = modinv((s // g) % step, step)
                        residue = ((targ // g) * inv) % step

                for da, db, dc in cs:
                    # k range so that each side has the chosen digit length
                    L = ceil_div(p[da - 1], a)
                    R = (p[da] - 1) // a
                    if L > R:
                        continue
                    L2 = ceil_div(p[db - 1], b)
                    R2 = (p[db] - 1) // b
                    if L2 > R2:
                        continue
                    if L2 > L:
                        L = L2
                    if R2 < R:
                        R = R2
                    if L > R:
                        continue
                    L3 = ceil_div(p[dc - 1], c)
                    R3 = (p[dc] - 1) // c
                    if L3 > R3:
                        continue
                    if L3 > L:
                        L = L3
                    if R3 < R:
                        R = R3
                    if L > R:
                        continue
                    if L < 1:
                        L = 1

                    # Align to the required congruence class (or take all k if step=1).
                    if step == 1:
                        k = L
                    else:
                        rem = L % step
                        if rem <= residue:
                            k = L + (residue - rem)
                        else:
                            k = L + (step - (rem - residue))

                    A = a * k
                    B = b * k
                    C = c * k
                    incA, incB, incC = a * step, b * step, c * step

                    while k <= R:
                        ma = mask_fixed_len(A, da, base, t)
                        if ma != -1:
                            mb = mask_fixed_len(B, db, base, t)
                            if mb != -1 and (ma & mb) == 0:
                                mc = mask_fixed_len(C, dc, base, t)
                                if (
                                    mc != -1
                                    and (ma & mc) == 0
                                    and (mb & mc) == 0
                                    and (ma | mb | mc) == fm
                                ):
                                    sums[base] += A

                        k += step
                        A += incA
                        B += incB
                        C += incC

    return sum(sums[base] for base in range(9, 19))


if __name__ == "__main__":
    print(solve())
