#!/usr/bin/env python
"""
Project Euler 507: Shortest Lattice Vector

Let V_n and W_n be the two lattice basis vectors defined from the tribonacci residues r_i in the problem
statement. For each n, define S(n) as the minimum Manhattan length (L1 norm) among all non-zero lattice
vectors k*V_n + l*W_n with k,l integers.

This program computes:
    sum_{n=1..20_000_000} S(n)

No external libraries are used.
"""

MOD = 10_000_000
N = 20_000_000


def _best_reduction(
    a1: int, a2: int, a3: int, b1: int, b2: int, b3: int, nb: int
) -> tuple[int, int]:
    """
    Return the best integer m for B - m*A, and the resulting L1 norm.

    The norm as a function of real m is convex and piecewise linear.  Its
    slopes can only change at b_i / a_i, so an integer minimizer is found by
    testing the two neighboring integers for each defined coordinate ratio.
    """
    best_m = 0
    best_v = nb

    if a1:
        q = b1 // a1
        v = abs(b1 - q * a1) + abs(b2 - q * a2) + abs(b3 - q * a3)
        if v < best_v:
            best_v = v
            best_m = q
        q += 1
        v = abs(b1 - q * a1) + abs(b2 - q * a2) + abs(b3 - q * a3)
        if v < best_v:
            best_v = v
            best_m = q

    if a2:
        q = b2 // a2
        v = abs(b1 - q * a1) + abs(b2 - q * a2) + abs(b3 - q * a3)
        if v < best_v:
            best_v = v
            best_m = q
        q += 1
        v = abs(b1 - q * a1) + abs(b2 - q * a2) + abs(b3 - q * a3)
        if v < best_v:
            best_v = v
            best_m = q

    if a3:
        q = b3 // a3
        v = abs(b1 - q * a1) + abs(b2 - q * a2) + abs(b3 - q * a3)
        if v < best_v:
            best_v = v
            best_m = q
        q += 1
        v = abs(b1 - q * a1) + abs(b2 - q * a2) + abs(b3 - q * a3)
        if v < best_v:
            best_v = v
            best_m = q

    return best_m, best_v


def _shortest_l1(v1: int, v2: int, v3: int, w1: int, w2: int, w3: int) -> int:
    """
    Compute S(n) for a single pair of basis vectors using 2D lattice reduction in the L1 norm.
    """
    a1, a2, a3 = v1, v2, v3
    b1, b2, b3 = w1, w2, w3

    # Degenerate safety (extremely unlikely for this construction, but harmless).
    if (a1 | a2 | a3) == 0:
        return abs(b1) + abs(b2) + abs(b3)
    if (b1 | b2 | b3) == 0:
        return abs(a1) + abs(a2) + abs(a3)

    # L1-analog of Lagrange/Minkowski reduction for rank-2 lattices.
    while True:
        na = abs(a1) + abs(a2) + abs(a3)
        nb = abs(b1) + abs(b2) + abs(b3)

        if nb < na:
            a1, a2, a3, b1, b2, b3 = b1, b2, b3, a1, a2, a3
            na, nb = nb, na

        m, reduced = _best_reduction(a1, a2, a3, b1, b2, b3, nb)
        if reduced >= nb:
            break

        b1 -= m * a1
        b2 -= m * a2
        b3 -= m * a3

    s = abs(a1) + abs(a2) + abs(a3)
    t = abs(b1) + abs(b2) + abs(b3)
    if t and t < s:
        s = t

    return s


def solve(limit: int = N) -> int:
    if limit <= 0:
        return 0

    # Stream the residues in 12-value blocks.  The first emitted value is r1,
    # and the recurrence is seeded by r_{-1}=1, r_0=0, r_1=0.
    prev2, prev1, cur = 1, 0, 0
    block = [0] * 12
    pos = 0
    count = 0
    total = 0
    sum10 = 0
    s1 = 0

    # Each next residue is below 3*MOD, so two subtractions replace modulo.
    for _ in range(limit * 12):
        block[pos] = cur
        pos += 1

        if pos == 12:
            s = _shortest_l1(
                block[0] - block[1],
                block[2] + block[3],
                block[4] * block[5],
                block[6] - block[7],
                block[8] + block[9],
                block[10] * block[11],
            )
            total += s
            count += 1
            if count == 1:
                s1 = s
            if count <= 10:
                sum10 += s
            pos = 0

        nxt = prev2 + prev1 + cur
        if nxt >= MOD:
            nxt -= MOD
            if nxt >= MOD:
                nxt -= MOD
        prev2, prev1, cur = prev1, cur, nxt

    # Problem statement checks
    assert s1 == 32
    if limit >= 10:
        assert sum10 == 130762273722

    return total


if __name__ == "__main__":
    print(solve())
