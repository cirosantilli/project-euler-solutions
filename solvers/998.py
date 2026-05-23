#!/usr/bin/env python
from math import gcd, isqrt


def pythagorean_partners(limit):
    """partners[m] contains every x with 0 < x <= m and m^2 + x^2 square."""
    partners = [[] for _ in range(limit + 1)]

    # Primitive triples are (r^2 - s^2, 2rs, r^2 + s^2), gcd(r,s)=1,
    # and r-s odd.  If both legs are <= limit, then the Euclid parameter r is
    # certainly below sqrt(2*limit); this integer bound is deliberately safe.
    for r in range(2, isqrt(2 * limit) + 3):
        rr = r * r
        for s in range(1, r):
            if ((r - s) & 1) == 0 or gcd(r, s) != 1:
                continue
            a = rr - s * s
            b = 2 * r * s
            m = a if a > b else b
            x = b if a > b else a
            if m > limit:
                continue
            for km in range(m, limit + 1, m):
                partners[km].append((km // m) * x)

    for row in partners:
        row.sort()
    return partners


def is_minimum_square(sides, twice_area, square_side):
    """Exact check that a triangle's minimum bounding square has this side."""
    ss = sides
    m = square_side
    m2 = m * m
    d_area = twice_area
    has_equal_candidate = False

    # Candidate 1: a side of the square is parallel to a side of the triangle.
    # For base d and adjacent sides e, f, the foot of the altitude is at
    # t = (d^2 + e^2 - f^2)/(2d).  The enclosing rectangle has width equal to
    # the span of {0, d, t} and height D/d, where D is twice the area.
    for i, d in enumerate(ss):
        e = ss[(i + 1) % 3]
        f = ss[(i + 2) % 3]
        den = 2 * d
        t_num = d * d + e * e - f * f
        d_num = d * den
        width_num = max(0, d_num, t_num) - min(0, d_num, t_num)

        width_cmp = width_num - m * den
        height_cmp = d_area - m * d
        if width_cmp < 0 and height_cmp < 0:
            return False
        if width_cmp <= 0 and height_cmp <= 0 and (width_cmp == 0 or height_cmp == 0):
            has_equal_candidate = True

    # Candidate 2: all four sides of the square are touched.  At a triangle
    # vertex with adjacent sides p, q and opposite side r, let
    # K = p q cos(theta) and D = p q sin(theta).  The balanced square satisfies
    # B^2 = K^2 / (p^2 + q^2 - 2D).  All comparisons below are integer-only.
    for i, r in enumerate(ss):
        p = ss[(i + 1) % 3]
        q = ss[(i + 2) % 3]
        k_num = p * p + q * q - r * r  # 2K
        if k_num <= 0:
            continue
        r_den_part = p * p + q * q - 2 * d_area
        if r_den_part <= 0:
            continue

        num = k_num * k_num  # numerator of 4B^2
        den = 4 * r_den_part  # denominator of B^2

        # Validity of the four-sided contact pattern: the two adjacent sides
        # must hit the far sides of the square, not miss them or pass a corner.
        p2 = p * p
        q2 = q * q
        if num < d_area * den:
            continue
        if num > p2 * den or num > q2 * den:
            continue
        if p2 * den > 2 * num or q2 * den > 2 * num:
            continue

        target = m2 * den
        if num < target:
            return False
        if num == target:
            has_equal_candidate = True

    return has_equal_candidate


def solve(limit):
    partners = pythagorean_partners(limit)
    seen = set()
    total = 0

    for m in range(1, limit + 1):
        mm = m * m
        row = [(0, m)]
        for x in partners[m]:
            row.append((x, isqrt(mm + x * x)))

        # Edge-aligned minima.  Put the altitude foot between the base endpoints:
        # vertices are (-x, 0), (y, 0), (0, m), so the side lengths are
        # x+y, sqrt(m^2+x^2), sqrt(m^2+y^2).  The bounding square has side m
        # exactly when x+y <= m; it is globally minimal iff the balanced square
        # at the top vertex is not smaller, giving xy >= m(m-x-y).
        row_len = len(row)
        for i, (x, hx) in enumerate(row):
            for j in range(i, row_len):
                y, hy = row[j]
                base = x + y
                if base == 0:
                    continue
                if base > m:
                    break
                if x * y < m * (m - base):
                    continue
                sides = tuple(sorted((base, hx, hy)))
                if sides not in seen:
                    seen.add(sides)
                    total += sides[0] + sides[1] + sides[2]

        # Four-sided, balanced minima.  In a square of side m, use vertices
        # (0,0), (m,u), (v,m).  Then m^2+u^2 and m^2+v^2 must be squares, and
        # so must (m-u)^2 + (m-v)^2.
        for i, (u, hu) in enumerate(row):
            for j in range(i, row_len):
                v, hv = row[j]
                twice_area = mm - u * v
                if twice_area <= 0:
                    continue
                p = m - u
                q = m - v
                third2 = p * p + q * q
                third = isqrt(third2)
                if third * third != third2 or third == 0:
                    continue
                sides = tuple(sorted((third, hu, hv)))
                if sides in seen:
                    continue
                if is_minimum_square(sides, twice_area, m):
                    seen.add(sides)
                    total += sides[0] + sides[1] + sides[2]

    return total


if __name__ == "__main__":
    assert solve(40) == 346
    assert solve(400) == 76402
    assert solve(2000) == 3237036
    print(solve(10**6))
