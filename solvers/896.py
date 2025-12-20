#!/usr/bin/env python3
"""
Project Euler 896 — Divisible Ranges

A range [a..a+L-1] is "divisible" if its numbers can be permuted so that
the n-th term is a multiple of n, for n = 1..L.

This program computes the 36th divisible range of length 36 and prints
the smallest number a.

No external libraries are used.
"""

import math
from typing import Dict, List, Optional, Set, Tuple


def egcd(a: int, b: int) -> Tuple[int, int, int]:
    """Extended gcd: returns (g, x, y) with ax + by = g = gcd(a,b)."""
    x0, y0, x1, y1 = 1, 0, 0, 1
    while b:
        q = a // b
        a, b = b, a % b
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    return a, x0, y0


def crt_merge(r1: int, m1: int, r2: int, m2: int) -> Optional[Tuple[int, int]]:
    """
    Merge:
        a ≡ r1 (mod m1)
        a ≡ r2 (mod m2)
    Returns (r, m) with a ≡ r (mod m), where m = lcm(m1, m2),
    or None if inconsistent.
    """
    g = math.gcd(m1, m2)
    if (r2 - r1) % g != 0:
        return None
    l = (m1 // g) * m2
    # Reduce to:
    #   r = r1 + m1 * t
    #   m1 * t ≡ (r2 - r1) (mod m2)
    m1g = m1 // g
    m2g = m2 // g
    diff = (r2 - r1) // g
    _, inv, _ = egcd(m1g, m2g)
    inv %= m2g
    t = (diff * inv) % m2g
    r = (r1 + m1 * t) % l
    return r, l


def lcm_upto(n: int) -> int:
    m = 1
    for i in range(1, n + 1):
        m = (m // math.gcd(m, i)) * i
    return m


def _candidates_offsets(L: int, unused_mask: int, target: int, step: int) -> List[int]:
    """
    Return offsets j in [0..L-1] such that:
      j ≡ target (mod step)
    and j is unused (bit set in unused_mask).
    """
    res: List[int] = []
    j = target
    while j < L:
        if (unused_mask >> j) & 1:
            res.append(j)
        j += step
    return res


def _pick_index_mrv(
    L: int, r: int, m: int, unused_mask: int, remaining_mask: int
) -> Optional[Tuple[int, List[int]]]:
    """
    Choose the remaining index i with the fewest feasible offsets, i.e. MRV.
    Tie-break towards larger i (important to constrain the CRT early).
    """
    best_i = 0
    best_cands: Optional[List[int]] = None
    best_count = 10**9

    # Iterate indices descending to get good tie-breaking for free.
    for i in range(L, 0, -1):
        if ((remaining_mask >> (i - 1)) & 1) == 0:
            continue
        g = math.gcd(m, i)
        target = (-r) % g
        cands = _candidates_offsets(L, unused_mask, target, g)
        c = len(cands)
        if c == 0:
            return None
        if c < best_count or (c == best_count and i > best_i):
            best_count = c
            best_i = i
            best_cands = cands

    return best_i, (best_cands if best_cands is not None else [])


def enumerate_valid_residues(L: int) -> Tuple[Set[int], int]:
    """
    Enumerate all residues a (mod M) for which [a..a+L-1] is a divisible range,
    where M = lcm(1..L).

    Returns (residue_set, M).

    Key idea:
      Choosing which offset j hosts the multiple of i forces a ≡ -j (mod i).
      A full permutation corresponds to a consistent system of congruences.
      We enumerate those systems with backtracking + CRT, using bitmasks and
      memoization to keep the state space small.
    """
    M = lcm_upto(L)

    all_offsets_mask = (1 << L) - 1
    all_indices_mask = (1 << L) - 1  # bit (i-1) => index i remaining

    residues: Set[int] = set()
    visited: Set[Tuple[int, int, int, int]] = set()

    def dfs(r: int, m: int, unused_mask: int, remaining_mask: int) -> None:
        r %= m
        key = (r, m, unused_mask, remaining_mask)
        if key in visited:
            return
        visited.add(key)

        if remaining_mask == 0:
            # m should have reached M; store the representative residue in [0..m-1]
            residues.add(r)
            return

        pick = _pick_index_mrv(L, r, m, unused_mask, remaining_mask)
        if pick is None:
            return
        i, cands = pick
        remaining2 = remaining_mask & ~(1 << (i - 1))

        for j in cands:
            merged = crt_merge(r, m, (-j) % i, i)
            if merged is None:
                continue
            r2, m2 = merged
            dfs(r2, m2, unused_mask & ~(1 << j), remaining2)

    dfs(0, 1, all_offsets_mask, all_indices_mask)
    return residues, M


def nth_divisible_range_start(L: int, n: int) -> int:
    """
    Return the n-th divisible range start a for length L (1-indexed).
    """
    residues, M = enumerate_valid_residues(L)
    starts: List[int] = []
    for r in residues:
        starts.append(r if r > 0 else M)  # a must be positive
    starts.sort()
    return starts[n - 1]


def is_divisible_range(a: int, L: int) -> bool:
    """
    Verify divisibility of a specific range [a..a+L-1] via bipartite matching.

    Left nodes: indices 1..L
    Right nodes: offsets 0..L-1 representing numbers a+offset
    Edge i->j if (a+j) is divisible by i.
    """
    # Build adjacency lists (small: L<=36 in this problem).
    adj: List[List[int]] = [[] for _ in range(L + 1)]
    for i in range(1, L + 1):
        row = []
        for j in range(L):
            if (a + j) % i == 0:
                row.append(j)
        adj[i] = row

    match_r = [-1] * L  # which i is matched to offset j

    def dfs(i: int, seen: List[bool]) -> bool:
        for j in adj[i]:
            if seen[j]:
                continue
            seen[j] = True
            if match_r[j] == -1 or dfs(match_r[j], seen):
                match_r[j] = i
                return True
        return False

    for i in range(1, L + 1):
        seen = [False] * L
        if not dfs(i, seen):
            return False
    return True


def _self_test() -> None:
    # Tests from the problem statement (length 4):
    # The first three divisible ranges of length 4 are [1..4], [2..5], [3..6],
    # and [6..9] is the 4th such range.
    assert nth_divisible_range_start(4, 1) == 1
    assert nth_divisible_range_start(4, 2) == 2
    assert nth_divisible_range_start(4, 3) == 3
    assert nth_divisible_range_start(4, 4) == 6
    assert is_divisible_range(6, 4)


def main() -> None:
    _self_test()
    print(nth_divisible_range_start(36, 36))


if __name__ == "__main__":
    main()
