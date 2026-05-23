#!/usr/bin/env python
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


def _gen_values_u32(n: int) -> array:
    """Generate a_1..a_n as unsigned 32-bit values."""
    vals = array("I")
    a = BASE % P
    for _ in range(n):
        vals.append(a)
        a = (a * BASE) % P
    return vals


def _rank_lookup(vals: array) -> tuple[array, int]:
    """Return value -> compressed rank lookup and the number of ranks."""
    seen = bytearray(P)
    for value in vals:
        seen[value] = 1

    rank_of = array("I", [0]) * P
    rank = 0
    for value, is_present in enumerate(seen):
        if is_present:
            rank += 1
            rank_of[value] = rank

    return rank_of, rank


def compute_S_mod(n: int) -> int:
    """
    Compute S(n) modulo MOD for subsequences of length 4.

    Uses DP over increasing subsequences with Fenwick trees over value ranks:
      For each length L we track (count_L, sum_L) of all increasing subsequences
      of length L ending at each value.
    """
    vals = _gen_values_u32(n)
    rank_of, m = _rank_lookup(vals)

    bit1_count = array("I", [0]) * (m + 1)
    bit1_sum = array("I", [0]) * (m + 1)
    bit2_count = array("I", [0]) * (m + 1)
    bit2_sum = array("I", [0]) * (m + 1)
    bit3_count = array("I", [0]) * (m + 1)
    bit3_sum = array("I", [0]) * (m + 1)

    total = 0
    mod = MOD

    for x in vals:
        rank = rank_of[x]
        query_rank = rank - 1

        count3 = sum3 = 0
        i = query_rank
        while i:
            count3 += bit3_count[i]
            sum3 += bit3_sum[i]
            i -= i & -i
        count3 %= mod
        sum3 %= mod
        total = (total + sum3 + count3 * x) % mod

        count2 = sum2 = 0
        i = query_rank
        while i:
            count2 += bit2_count[i]
            sum2 += bit2_sum[i]
            i -= i & -i
        count2 %= mod
        sum2 %= mod

        delta_count = count2
        delta_sum = (sum2 + count2 * x) % mod
        i = rank
        while i <= m:
            value = bit3_count[i] + delta_count
            if value >= mod:
                value -= mod
            bit3_count[i] = value

            value = bit3_sum[i] + delta_sum
            if value >= mod:
                value -= mod
            bit3_sum[i] = value
            i += i & -i

        count1 = sum1 = 0
        i = query_rank
        while i:
            count1 += bit1_count[i]
            sum1 += bit1_sum[i]
            i -= i & -i
        count1 %= mod
        sum1 %= mod

        delta_count = count1
        delta_sum = (sum1 + count1 * x) % mod
        i = rank
        while i <= m:
            value = bit2_count[i] + delta_count
            if value >= mod:
                value -= mod
            bit2_count[i] = value

            value = bit2_sum[i] + delta_sum
            if value >= mod:
                value -= mod
            bit2_sum[i] = value
            i += i & -i

        i = rank
        while i <= m:
            value = bit1_count[i] + 1
            if value >= mod:
                value -= mod
            bit1_count[i] = value

            value = bit1_sum[i] + x
            if value >= mod:
                value -= mod
            bit1_sum[i] = value
            i += i & -i

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
