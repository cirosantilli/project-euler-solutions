#!/usr/bin/env python3
"""
Project Euler 733: Ascending Subsequences

Compute S(10^6) mod 1_000_000_007 for the sequence:
    a_i = 153^i mod 10_000_019, i >= 1

S(n) is the sum of the terms over all strictly increasing subsequences of length 4
chosen from the first n terms of a_i.

No external libraries are used (only Python standard library).
"""

from array import array

P = 10_000_019
BASE = 153
MOD = 1_000_000_007

# Pack (count, sum) into one 64-bit integer to halve Fenwick traversals.
SHIFT = 32
MASK = (1 << SHIFT) - 1


def _gen_values_u32(n: int) -> array:
    """Generate a_1..a_n as unsigned 32-bit values."""
    vals = array("I")
    a = BASE % P
    for _ in range(n):
        vals.append(a)
        a = (a * BASE) % P
    return vals


def _compress_to_ranks(vals: array) -> tuple[array, int]:
    """
    Coordinate-compress vals to ranks 1..m (stable under < comparison).
    Returns (ranks, m).
    """
    n = len(vals)
    idx = list(range(n))
    idx.sort(key=vals.__getitem__)

    ranks = array("I", [0]) * n
    rank = 0
    prev = None
    for i in idx:
        v = vals[i]
        if v != prev:
            rank += 1
            prev = v
        ranks[i] = rank

    return ranks, rank


def _fenwick_sum_packed(bit: array, i: int) -> tuple[int, int]:
    """Return (count, sum) prefix up to i (both mod MOD), packed BIT."""
    c = 0
    s = 0
    while i:
        p = bit[i]
        c += p >> SHIFT
        s += p & MASK
        i -= i & -i
    return c % MOD, s % MOD


def _fenwick_add_packed(bit: array, n: int, i: int, dc: int, ds: int) -> None:
    """Add (dc, ds) (mod MOD) at position i, packed BIT."""
    # Ensure deltas are in [0, MOD).
    dc %= MOD
    ds %= MOD

    while i <= n:
        p = bit[i]
        c = (p >> SHIFT) + dc
        if c >= MOD:
            c -= MOD
        s = (p & MASK) + ds
        if s >= MOD:
            s -= MOD
        bit[i] = (c << SHIFT) | s
        i += i & -i


def compute_S_mod(n: int) -> int:
    """
    Compute S(n) modulo MOD for subsequences of length 4.

    Uses DP over increasing subsequences with Fenwick trees over value ranks:
      For each length L we track (count_L, sum_L) of all increasing subsequences
      of length L ending at each value.
    """
    vals = _gen_values_u32(n)
    ranks, m = _compress_to_ranks(vals)

    # Three packed Fenwick trees for lengths 1,2,3.
    bit1 = array("Q", [0]) * (m + 1)
    bit2 = array("Q", [0]) * (m + 1)
    bit3 = array("Q", [0]) * (m + 1)

    total = 0

    # Local bindings for speed.
    fen_sum = _fenwick_sum_packed
    fen_add = _fenwick_add_packed
    mod = MOD

    for x, r in zip(vals, ranks):
        x_mod = int(x)  # < MOD already
        rm1 = int(r) - 1

        # Length 4 contribution ending here:
        c3, s3 = fen_sum(bit3, rm1)
        total = (total + s3 + (c3 * x_mod) % mod) % mod

        # Build length 3 subsequences ending here from length 2:
        c2, s2 = fen_sum(bit2, rm1)
        fen_add(bit3, m, int(r), c2, s2 + (c2 * x_mod) % mod)

        # Build length 2 subsequences ending here from length 1:
        c1, s1 = fen_sum(bit1, rm1)
        fen_add(bit2, m, int(r), c1, s1 + (c1 * x_mod) % mod)

        # Length 1 subsequence (the element alone):
        fen_add(bit1, m, int(r), 1, x_mod)

    return total


# --- Exact (non-mod) checker for the small test values from the statement ---


def compute_S_exact(n: int) -> int:
    """
    Compute exact S(n) (no modulus), intended for small n (e.g. n<=100).
    """
    vals = [int(x) for x in _gen_values_u32(n)]
    # Coordinate compression.
    idx = list(range(n))
    idx.sort(key=vals.__getitem__)
    ranks = [0] * n
    rank = 0
    prev = None
    for i in idx:
        v = vals[i]
        if v != prev:
            rank += 1
            prev = v
        ranks[i] = rank

    m = rank

    def add(bit, i, delta):
        while i <= m:
            bit[i] += delta
            i += i & -i

    def pref(bit, i):
        s = 0
        while i:
            s += bit[i]
            i -= i & -i
        return s

    b1c = [0] * (m + 1)
    b1s = [0] * (m + 1)
    b2c = [0] * (m + 1)
    b2s = [0] * (m + 1)
    b3c = [0] * (m + 1)
    b3s = [0] * (m + 1)

    total = 0
    for x, r in zip(vals, ranks):
        c3 = pref(b3c, r - 1)
        s3 = pref(b3s, r - 1)
        total += s3 + c3 * x

        c2 = pref(b2c, r - 1)
        s2 = pref(b2s, r - 1)
        add(b3c, r, c2)
        add(b3s, r, s2 + c2 * x)

        c1 = pref(b1c, r - 1)
        s1 = pref(b1s, r - 1)
        add(b2c, r, c1)
        add(b2s, r, s1 + c1 * x)

        add(b1c, r, 1)
        add(b1s, r, x)

    return total


def main() -> None:
    # Test values given in the problem statement.
    assert compute_S_exact(6) == 94513710
    assert compute_S_exact(100) == 4465488724217

    print(compute_S_mod(1_000_000))


if __name__ == "__main__":
    main()
