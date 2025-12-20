#!/usr/bin/env python3
"""Project Euler 941: de Bruijn's Combination Lock

This program computes F(N) as defined in the problem statement, and prints
F(10^7) mod 1234567891 by default.

Rules satisfied:
- No external libraries (only Python standard library).
- Asserts are included for the example values given in the statement.

Implementation overview (high-level):
- Rank each 12-digit string by its first occurrence position in the
  lexicographically smallest de Bruijn sequence C(10,12).
- Sort the N strings by that rank using a radix sort (since N can be 10^7).
- Compute the required weighted sum modulo 1234567891.

The ranking routine is a careful Python port of the practical RankDB algorithm
from Sawada & Williams ("Practical Algorithms to Rank Necklaces, Lyndon Words,
and de Bruijn Sequences"), specialized to small n=12.
"""

from __future__ import annotations

import sys
from array import array


# -------------------------
# Small de Bruijn generator
# -------------------------


def de_bruijn_sequence(k: int, n: int) -> str:
    """Return the *linearized* lexicographically smallest de Bruijn sequence C(k,n).

    Output length is k**n + n - 1 and begins with '0'*n.

    This is used only for small self-tests (e.g. C(3,2) from the statement).
    """

    # Standard recursive construction that outputs a cyclic de Bruijn sequence
    # in lexicographically smallest order.
    a = [0] * (k * n)
    seq: list[int] = []

    def db(t: int, p: int) -> None:
        if t > n:
            if n % p == 0:
                seq.extend(a[1 : p + 1])
        else:
            a[t] = a[t - p]
            db(t + 1, p)
            start = a[t - p] + 1
            for j in range(start, k):
                a[t] = j
                db(t + 1, t)

    db(1, 1)
    s = "".join(str(x) for x in seq)
    return s + s[: n - 1]


# -------------------------------------
# Rank substrings in lexicographic DB(n)
# -------------------------------------


def make_rank_db(k: int, n: int):
    """Create a fast RankDB(w) function for fixed (k,n).

    The returned function expects w as a list of length n+1 using indices 1..n,
    with symbols in {1,2,...,k}. It returns a 1-based index r such that the
    length-n substring starting at position r in the linearized lexicographically
    smallest de Bruijn sequence DB(n) is exactly w.
    """

    # power[i] = k**i
    power = [1] * (n + 1)
    for i in range(1, n + 1):
        power[i] = power[i - 1] * k

    # Scratch buffers reused across calls to avoid per-call allocations.
    neck_rep = [0] * (n + 1)  # necklace representative of w
    prev = [0] * (n + 1)  # helper for LargestNecklace

    # Scratch used inside T (kept separate so RankDB's neck_rep isn't clobbered).
    t_neck = [0] * (n + 1)
    B = [[0] * (n + 1) for _ in range(n + 1)]
    suf = [[0] * (n + 1) for _ in range(n + 1)]

    def lyn(w: list[int]) -> int:
        p = 1
        # Find length of the longest prefix of w[1..n] that is a Lyndon word.
        for i in range(2, n + 1):
            wi = w[i]
            wip = w[i - p]
            if wi < wip:
                return p
            if wi > wip:
                p = i
        return p

    def is_necklace(w: list[int]) -> bool:
        p = 1
        for i in range(2, n + 1):
            wi = w[i]
            wip = w[i - p]
            if wi < wip:
                return False
            if wi > wip:
                p = i
        return (n % p) == 0

    def largest_necklace(w: list[int], out: list[int]) -> None:
        # out[1..n] = largest necklace <= w[1..n]
        for i in range(1, n + 1):
            out[i] = w[i]
        while not is_necklace(out):
            p = lyn(out)
            out[p] -= 1
            for i in range(p + 1, n + 1):
                out[i] = k

    def T(w: list[int]) -> int:
        """Return the number of strings whose necklace is <= w[1..n]."""

        largest_necklace(w, t_neck)

        # Compute B[t][j]. Only indices 0..t are meaningful.
        B[0][0] = 1
        for t in range(1, n + 1):
            Bt = B[t]
            Bt[t] = 0
            # Note: B[t-j-1][0] is always defined because 0 <= j < t.
            for j in range(t - 1, -1, -1):
                Bt[j] = Bt[j + 1] + (k - t_neck[j + 1]) * B[t - j - 1][0]

        # Compute suf[i][j] for i>=2.
        for i in range(2, n + 1):
            s = i
            for j in range(i, n + 1):
                # If t_neck[j] > t_neck[j-s+1], update s.
                if t_neck[j] > t_neck[j - s + 1]:
                    s = j + 1
                suf[i][j] = j - s + 1

        tot = lyn(t_neck)
        for t in range(1, n + 1):
            b0 = B[t - 1][0]
            for j in range(0, n):
                if j + t <= n:
                    tot += b0 * (t_neck[j + 1] - 1) * power[n - t - j]
                else:
                    # s is the length of the longest suffix of t_neck[n-t+2..j]
                    # that is a prefix of t_neck[1..n].
                    if j < n - t + 2:
                        sfx = 0
                    else:
                        sfx = suf[n - t + 2][j]
                    if t_neck[j + 1] > t_neck[sfx + 1]:
                        tot += (
                            B[n - j + sfx][sfx + 1]
                            + (t_neck[j + 1] - t_neck[sfx + 1] - 1) * B[n - j - 1][0]
                        )

        return tot

    def rank_db(w: list[int]) -> int:
        """Return the 1-based rank (start position) of w in DB(n)."""

        # Special case: w = k^t 1^(n-t) for t>=1 (wraparound substrings).
        t = 0
        while t < n and w[t + 1] == k:
            t += 1
        j = t
        while j < n and w[j + 1] == 1:
            j += 1
        if t >= 1 and j == n:
            return power[n] - t + 1

        # If w is already a necklace, we're done.
        if is_necklace(w):
            return 1 - lyn(w) + T(w)

        # Find the necklace representative of w and its rotation offset s.
        for i in range(1, n + 1):
            neck_rep[i] = w[i]
        s = 0
        while not is_necklace(neck_rep):
            s += 1
            # Rotate w left by s positions.
            for i in range(1, n + 1):
                j2 = i + s
                neck_rep[i] = w[j2] if j2 <= n else w[j2 - n]

        # Recurrence cases (note: neck_rep is a necklace).
        if s != t:
            return rank_db(neck_rep) + lyn(neck_rep) - s
        if lyn(neck_rep) < n:
            return rank_db(neck_rep) - s

        # Otherwise, adjust suffix to 1s, move to previous necklace.
        for i in range(n - s + 1, n + 1):
            neck_rep[i] = 1
        largest_necklace(neck_rep, prev)
        return rank_db(prev) + lyn(prev) - s

    return rank_db


# ---------------------------
# LCG + digit conversion (12)
# ---------------------------

MOD_1E12 = 10**12
LCG_MUL = 920461
LCG_ADD = 800217387569


def make_digits4_table() -> list[tuple[int, int, int, int]]:
    """digits4[x] gives the 4 base-10 digits of x (0..9999), each shifted by +1.

    We shift by +1 because RankDB expects alphabet symbols in {1..k}.
    """

    out: list[tuple[int, int, int, int]] = [None] * 10000  # type: ignore
    for x in range(10000):
        a = x // 1000
        b = (x // 100) % 10
        c = (x // 10) % 10
        d = x % 10
        out[x] = (a + 1, b + 1, c + 1, d + 1)
    return out


_DIG4 = make_digits4_table()


def int_to_word12_symbols_1_to_10(x: int, w: list[int]) -> None:
    """Fill w[1..12] with the 12 digits of x in base 10 (with leading zeros),
    mapped to symbols 1..10.

    w must be length >= 13.
    """

    # Split into 3 chunks of 4 digits to avoid expensive string formatting.
    hi = x // 100_000_000  # 10^8
    mid = (x // 10_000) % 10_000
    lo = x % 10_000

    d0 = _DIG4[hi]
    d1 = _DIG4[mid]
    d2 = _DIG4[lo]

    w[1], w[2], w[3], w[4] = d0
    w[5], w[6], w[7], w[8] = d1
    w[9], w[10], w[11], w[12] = d2


def generate_a_values(N: int):
    """Yield a_1..a_N from the LCG (a_0 = 0)."""

    a = 0
    for _ in range(N):
        a = (LCG_MUL * a + LCG_ADD) % MOD_1E12
        yield a


# ----------------
# Sorting utilities
# ----------------


def radix_sort_pairs_u64(keys: array, vals: array, bits_per_pass: int = 16) -> None:
    """In-place stable radix sort of (keys, vals) by keys.

    keys and vals are array('Q') of equal length. On return, both are permuted
    so that keys is sorted non-decreasingly, and vals is permuted alongside.

    This uses LSD radix sort with radix 2**bits_per_pass.
    """

    n = len(keys)
    if n <= 1:
        return

    mask = (1 << bits_per_pass) - 1
    radix = 1 << bits_per_pass

    tmp_keys = array("Q", [0]) * n
    tmp_vals = array("Q", [0]) * n

    # Our ranks fit comfortably within 48 bits for k=10,n=12, so 3 passes of 16 bits suffice.
    # Using a fixed number of passes avoids scanning for max().
    for shift in (0, bits_per_pass, 2 * bits_per_pass):
        counts = [0] * radix

        for i in range(n):
            counts[(keys[i] >> shift) & mask] += 1

        total = 0
        for b in range(radix):
            c = counts[b]
            counts[b] = total
            total += c

        for i in range(n):
            x = keys[i]
            b = (x >> shift) & mask
            p = counts[b]
            tmp_keys[p] = x
            tmp_vals[p] = vals[i]
            counts[b] = p + 1

        keys, tmp_keys = tmp_keys, keys
        vals, tmp_vals = tmp_vals, vals

    # If we did an odd number of passes, data is now in the swapped arrays.
    # With 3 passes, keys/vals currently refer to the swapped local variables,
    # so we need to copy back to the original arrays.
    # After 3 swaps, the sorted data is in local variables keys/vals, not
    # necessarily in the original objects.
    # Detect this via identity check.
    #
    # Note: array('Q') doesn't support buffer swapping, so we copy.
    if keys is not tmp_keys:  # keys holds the sorted data
        # keys/vals are the sorted locals, tmp_keys/tmp_vals are the originals.
        tmp_keys[:] = keys
        tmp_vals[:] = vals


# ------------------------
# F(N) computation wrappers
# ------------------------


def compute_F_mod_large(N: int, rank_db, mod: int) -> int:
    """Compute F(N) mod mod for large N using radix sort."""

    w = [0] * 13
    keys = array("Q")
    vals = array("Q")
    keys_append = keys.append
    vals_append = vals.append

    for a in generate_a_values(N):
        int_to_word12_symbols_1_to_10(a, w)
        keys_append(rank_db(w))
        vals_append(a)

    radix_sort_pairs_u64(keys, vals)

    acc = 0
    for i in range(N):
        acc = (acc + (i + 1) * (vals[i] % mod)) % mod
    return acc


# ---------
# Self tests
# ---------


def _self_tests() -> None:
    # Statement's C(3,2)
    assert de_bruijn_sequence(3, 2) == "0010211220"

    # Sanity: RankDB must match direct positions for a tiny case.
    # k=2,n=4 has only 16 words.
    k2, n2 = 2, 4
    db2 = de_bruijn_sequence(k2, n2)
    r2 = make_rank_db(k2, n2)
    for x in range(k2**n2):
        s = format(x, f"0{n2}b")  # binary string
        w = [0] * (n2 + 1)
        for i, ch in enumerate(s, 1):
            w[i] = (ord(ch) - 48) + 1
        # find first occurrence in linearized DB
        idx = db2.find(s)
        assert idx >= 0
        assert r2(w) == idx + 1

    # Statement's numeric test values for this problem.
    rank_db_10_12 = make_rank_db(10, 12)
    assert compute_F_mod_large(2, rank_db_10_12, 2**64 - 1) == 2194210461325
    assert compute_F_mod_large(10, rank_db_10_12, 2**64 - 1) == 32698850376317


def main(argv: list[str]) -> None:
    _self_tests()
    N = 10_000_000
    if len(argv) >= 2:
        N = int(argv[1])
    mod = 1234567891
    rank_db = make_rank_db(10, 12)
    print(compute_F_mod_large(N, rank_db, mod))


if __name__ == "__main__":
    main(sys.argv)
