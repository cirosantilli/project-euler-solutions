#!/usr/bin/env python3
"""
Project Euler 867: Tiling Dodecagon

Compute T(10) modulo 1_000_000_007.

Approach (high level):
- Reduce tilings of a regular dodecagon to a recursion over smaller equiangular 12-gons.
- Each reduction step introduces 6 identical corner regions that can be tiled only with
  unit triangles and unit hexagons; rectangles are forced to be all squares (unique).
- Counting tilings of those corner regions (a truncated equilateral triangle) and of
  the central regular hexagon becomes a problem of selecting non-overlapping unit
  hexagons in a fixed triangle grid. This is equivalent to counting independent sets
  on a small “hexagon-center” lattice.
- Count independent sets with a row-by-row DP using bitmasks plus a subset-sum
  (zeta transform) speedup.

No external libraries are used.
"""

from functools import lru_cache

MOD = 1_000_000_007

# Precompute all bitmasks of length L with no adjacent 1-bits.
# (Independent sets on a simple path graph of length L.)
MAX_L = 2 * 10 - 1  # for n=10, the widest row length needed is 19
INDEP = [[] for _ in range(MAX_L + 1)]
for L in range(MAX_L + 1):
    masks = []
    for m in range(1 << L):
        if (m & (m << 1)) == 0:
            masks.append(m)
    INDEP[L] = masks


def _subset_sums(values, L):
    """
    Subset-sum (zeta) transform:
    out[mask] = sum(values[sub] for sub ⊆ mask)  (mod MOD)

    Runs in O(L * 2^L) using a block update pattern (no bit-tests).
    """
    out = values[:]  # copy
    size = 1 << L
    mod = MOD
    for i in range(L):
        step = 1 << i
        block = step << 1
        for start in range(0, size, block):
            mid = start + step
            end = start + block
            # out[mid:end] += out[start:mid]
            for m in range(mid, end):
                s = out[m] + out[m - step]
                out[m] = s - mod if s >= mod else s
    return out


@lru_cache(maxsize=None)
def _count_independent_sets(row_lengths):
    """
    Count independent sets on a layered lattice described by its row lengths.

    Representation:
    - Each row is a horizontal line of 'row_lengths[i]' nodes.
    - Inside a row, adjacent nodes are neighbors (so row masks must have no adjacent 1s).
    - Between consecutive rows, the neighbor relation depends on whether the next row
      is longer by 1 (growth) or shorter by 1 (shrink), matching a sheared embedding
      of the lattice:
        * If Ln = Lc + 1: forbid A overlapping B and B shifted right (B>>1).
        * If Ln = Lc - 1: forbid A overlapping B and B shifted left (B<<1).
      This exactly matches the “hexagon-center” lattice for these regions.

    The DP uses a subset-sum transform so that each step can sum all compatible
    previous-row masks in O(L * 2^L + count_masks_next).
    """
    if not row_lengths:
        return 1

    row_lengths = list(row_lengths)
    L0 = row_lengths[0]
    dp = [0] * (1 << L0)
    for m in INDEP[L0]:
        dp[m] = 1

    for i in range(1, len(row_lengths)):
        Lc = row_lengths[i - 1]
        Ln = row_lengths[i]
        if abs(Ln - Lc) != 1:
            raise ValueError("Unexpected row length change; expected ±1 steps only.")

        subs = _subset_sums(dp, Lc)
        dp2 = [0] * (1 << Ln)
        fullmask = (1 << Lc) - 1

        if Ln == Lc + 1:
            # Compatibility: A & (B | (B>>1)) == 0
            for b in INDEP[Ln]:
                forb = (b | (b >> 1)) & fullmask
                allowed = fullmask ^ forb
                dp2[b] = subs[allowed]
        else:
            # Ln == Lc - 1
            # Compatibility: A & (B | (B<<1)) == 0
            for b in INDEP[Ln]:
                forb = (b | (b << 1)) & fullmask
                allowed = fullmask ^ forb
                dp2[b] = subs[allowed]

        dp = dp2

    last_L = row_lengths[-1]
    total = 0
    for m in INDEP[last_L]:
        total += dp[m]
    return total % MOD


@lru_cache(maxsize=None)
def H(n):
    """
    Number of tilings of a regular hexagon of side n using unit triangles and unit hexagons.

    This reduces to counting independent sets on a hexagon-center lattice whose row lengths are:
    n, n+1, ..., 2n-1, 2n-2, ..., n
    """
    inc = list(range(n, 2 * n))
    dec = list(range(2 * n - 2, n - 1, -1))
    return _count_independent_sets(tuple(inc + dec))


@lru_cache(maxsize=None)
def F(n, h):
    """
    Number of tilings of a truncated equilateral triangle with base n and height h
    (in unit-edge layers), using unit triangles and unit hexagons.

    The corresponding hexagon-center lattice is a trapezoid with (h-1) rows whose lengths decrease by 1:
    (n-2), (n-3), ..., (n-h)
    (clamped at 0 for small parameters).
    """
    rows = h - 1
    lengths = tuple(max(0, (n - 2) - i) for i in range(rows))
    return _count_independent_sets(lengths)


@lru_cache(maxsize=None)
def R(u, v):
    """
    Helper recursion on equiangular 12-gons with alternating side lengths u and v.

    R(u,0) is the hexagon case.
    For v>0, peel off a layer of forced squares along the u-sides, leaving:
    - a smaller equiangular 12-gon with parameters (v,w)
    - six identical corner regions counted by F(u, u-w)
    The special +1 term handles the unique unit 12-gon tiling when (u,v)=(1,1).
    """
    if v == 0:
        return H(u)

    res = 1 if (u == 1 and v == 1) else 0
    for w in range(u):
        corner = F(u, u - w)
        res = (res + R(v, w) * pow(corner, 6, MOD)) % MOD
    return res


def T(n):
    """
    Number of tilings of a regular dodecagon of side n by unit regular polygons,
    returned modulo MOD.
    """
    ans = (2 * R(n, n) - (1 if n == 1 else 0)) % MOD
    return ans


def main():
    # Tests from the problem statement
    assert T(1) == 5
    assert T(2) == 48

    print(T(10))


if __name__ == "__main__":
    main()
