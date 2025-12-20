#!/usr/bin/env python3
"""Project Euler 868: Belfry Maths

The bell-ringers' procedure generates all permutations by adjacent swaps.
The number of swaps required to reach a given permutation from the
alphabetically-sorted start equals that permutation's 0-based index in
this generation order.

That generation order is the Steinhaus-Johnson-Trotter (SJT) order.
We compute the index (rank) directly, without simulating all permutations.

No external libraries are used.
"""

from __future__ import annotations

import sys
from typing import Dict, List


def sjt_rank(perm: List[int]) -> int:
    """Return the 0-based rank of `perm` in SJT order.

    `perm` must be a permutation of [1..n] for some n = len(perm).

    Recurrence (build-up view):
      SJT(n) is formed by taking the SJT(n-1) list in order. For each
      permutation with rank i in SJT(n-1), we insert n into it in k=n
      possible positions. The direction of insertion alternates with i:
        - i even: positions n-1, n-2, ..., 0 (right to left)
        - i odd : positions 0, 1, ..., n-1 (left to right)

    Reverse that to rank a given permutation:
      Let i be the rank of the permutation with n removed.
      Let p be the position of n in the current permutation (0 = leftmost).
      Then within its block:
        - if i even: t = (n-1-p)
        - if i odd : t = p
      rank = i*n + t
    """
    n = len(perm)
    if n <= 1:
        return 0

    # The elements must be exactly 1..n, so the maximum element is n.
    pos = perm.index(n)
    base = perm[:pos] + perm[pos + 1 :]
    i = sjt_rank(base)

    t = (n - 1 - pos) if (i % 2 == 0) else pos
    return i * n + t


def swaps_to_reach_word(word: str) -> int:
    """Number of swaps needed to reach `word` starting from sorted letters."""
    letters = list(word.strip())
    if not letters:
        raise ValueError("word must be non-empty")
    if len(set(letters)) != len(letters):
        raise ValueError("all letters must be distinct")

    sorted_letters = sorted(letters)
    mapping: Dict[str, int] = {ch: i + 1 for i, ch in enumerate(sorted_letters)}
    perm = [mapping[ch] for ch in letters]
    return sjt_rank(perm)


def _run_self_tests() -> None:
    # Test values given in the problem statement.
    assert swaps_to_reach_word("CBA") == 3
    assert swaps_to_reach_word("BELFRY") == 59


def main(argv: List[str]) -> None:
    _run_self_tests()

    target = "NOWPICKBELFRYMATHS"
    if len(argv) >= 2:
        target = argv[1].strip()

    print(swaps_to_reach_word(target))


if __name__ == "__main__":
    main(sys.argv)
