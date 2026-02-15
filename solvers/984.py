from __future__ import annotations

from fractions import Fraction
from typing import List

MOD: int = 1_000_000_007
TARGET_N: int = 10**18
SMALL_LIMIT: int = 20


def count_connected_canonical(width: int, height: int) -> int:
    """Count connected canonical octagonal shapes in a width x height box.

    Canonical means the shape touches all 4 sides of the bounding box and is
    representable as the intersection of:
    - x+y >= p
    - x+y <= width+height-2-q
    - x-y >= -(height-1)+c
    - x-y <= (width-1)-d
    with p,q,c,d >= 0.
    """
    n: int = width * height
    knight_dirs = (
        (2, 1),
        (2, -1),
        (-2, 1),
        (-2, -1),
        (1, 2),
        (1, -2),
        (-1, 2),
        (-1, -2),
    )

    # Bitmask of knight-neighbors for each cell index x*height + y.
    neighbor_masks: List[int] = [0] * n
    for x in range(width):
        for y in range(height):
            idx = x * height + y
            mask = 0
            for dx, dy in knight_dirs:
                nx = x + dx
                ny = y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    mask |= 1 << (nx * height + ny)
            neighbor_masks[idx] = mask

    # Prefix masks for each row x: row_prefix[x][y] has bits (x,0..y) set.
    row_prefix: List[List[int]] = [[0] * height for _ in range(width)]
    for x in range(width):
        base = x * height
        cur = 0
        row = row_prefix[x]
        for y in range(height):
            cur |= 1 << (base + y)
            row[y] = cur

    full: int = width + height - 2
    mlim: int = min(width - 1, height - 1)
    connected_count = 0

    for p in range(mlim + 1):
        for q in range(mlim + 1):
            s_upper = full - q
            for c in range(mlim + 1):
                d_lower = -(height - 1) + c
                for d in range(mlim + 1):
                    d_upper = (width - 1) - d

                    # Quick side-touch feasibility checks.
                    # Left side x=0
                    lo = max(0, p, -d_upper)
                    hi = min(height - 1, s_upper, -d_lower)
                    if lo > hi:
                        continue
                    # Right side x=width-1
                    lo = max(0, p - (width - 1), (width - 1) - d_upper)
                    hi = min(height - 1, s_upper - (width - 1), (width - 1) - d_lower)
                    if lo > hi:
                        continue
                    # Bottom side y=0
                    lo = max(0, p, d_lower)
                    hi = min(width - 1, s_upper, d_upper)
                    if lo > hi:
                        continue
                    # Top side y=height-1
                    lo = max(0, p - (height - 1), d_lower + (height - 1))
                    hi = min(width - 1, s_upper - (height - 1), d_upper + (height - 1))
                    if lo > hi:
                        continue

                    occ = 0
                    for x in range(width):
                        y_lo = p - x
                        t = x - d_upper
                        if t > y_lo:
                            y_lo = t
                        if y_lo < 0:
                            y_lo = 0

                        y_hi = s_upper - x
                        t = x - d_lower
                        if t < y_hi:
                            y_hi = t
                        if y_hi > height - 1:
                            y_hi = height - 1

                        if y_lo <= y_hi:
                            row_mask = row_prefix[x][y_hi]
                            if y_lo > 0:
                                row_mask ^= row_prefix[x][y_lo - 1]
                            occ |= row_mask

                    # Connectivity in knight graph using bitmask DFS.
                    if occ & (occ - 1) == 0:
                        connected_count += 1
                        continue

                    start = occ & -occ
                    seen = start
                    stack = [start.bit_length() - 1]

                    while stack:
                        u = stack.pop()
                        nbr = neighbor_masks[u] & occ & ~seen
                        while nbr:
                            lsb = nbr & -nbr
                            nbr ^= lsb
                            seen |= lsb
                            stack.append(lsb.bit_length() - 1)

                    if seen == occ:
                        connected_count += 1

    return connected_count


def compute_f_upto(limit: int) -> List[int]:
    """Compute f(n) exactly for all 1 <= n <= limit."""
    canonical_counts: List[List[int]] = [[0] * (limit + 1) for _ in range(limit + 1)]

    # Symmetry: count(width,height) == count(height,width)
    for width in range(1, limit + 1):
        for height in range(1, width + 1):
            val = count_connected_canonical(width, height)
            canonical_counts[width][height] = val
            canonical_counts[height][width] = val

    f: List[int] = [0] * (limit + 1)
    for n in range(1, limit + 1):
        total = 0
        for width in range(1, n + 1):
            wx = n - width + 1
            row = canonical_counts[width]
            for height in range(1, n + 1):
                total += wx * (n - height + 1) * row[height]
        f[n] = total
    return f


def lagrange_eval_mod(xs: List[int], ys: List[int], x: int, mod: int) -> int:
    """Evaluate interpolation polynomial at x in Z_mod."""
    x %= mod
    ans = 0
    n = len(xs)
    for i in range(n):
        xi = xs[i]
        num = 1
        den = 1
        for j in range(n):
            if i == j:
                continue
            xj = xs[j]
            num = (num * (x - xj)) % mod
            den = (den * (xi - xj)) % mod
        ans = (ans + ys[i] * num * pow(den, mod - 2, mod)) % mod
    return ans


def lagrange_eval_int(xs: List[int], ys: List[int], x: int) -> int:
    """Evaluate interpolation polynomial at x over Q, returning exact integer."""
    total = Fraction(0, 1)
    n = len(xs)
    for i in range(n):
        term = Fraction(ys[i], 1)
        xi = xs[i]
        for j in range(n):
            if i == j:
                continue
            xj = xs[j]
            term *= Fraction(x - xj, xi - xj)
        total += term
    if total.denominator != 1:
        raise ValueError("Expected integral interpolation value")
    return total.numerator


def solve() -> int:
    f = compute_f_upto(SMALL_LIMIT)

    # Given examples.
    assert f[3] == 9
    assert f[5] == 903

    # Even subsequence follows a degree-8 polynomial from n=4 onward.
    even_xs = [4, 6, 8, 10, 12, 14, 16, 18, 20]
    even_ys = [f[n] for n in even_xs]

    # Additional given checks.
    assert lagrange_eval_int(even_xs, even_ys, 100) == 8658918531876
    assert (
        lagrange_eval_mod(even_xs, [y % MOD for y in even_ys], 10000, MOD) == 377956308
    )

    return lagrange_eval_mod(even_xs, [y % MOD for y in even_ys], TARGET_N, MOD)


if __name__ == "__main__":
    print(solve())
