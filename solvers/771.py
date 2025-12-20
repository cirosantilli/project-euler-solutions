#!/usr/bin/env python3
"""
Project Euler 771 - Pseudo Geometric Sequences

Count all strictly increasing integer sequences a_0..a_n (n>=4) with
|a_i^2 - a_{i-1} a_{i+1}| <= 2, for max term <= N.
"""

from __future__ import annotations

from bisect import bisect_right
from typing import List

MOD = 1_000_000_007


def iroot4(n: int) -> int:
    x = int(n**0.25)
    while (x + 1) ** 4 <= n:
        x += 1
    while x**4 > n:
        x -= 1
    return x


def phi_sieve(n: int) -> List[int]:
    phi = list(range(n + 1))
    for i in range(2, n + 1):
        if phi[i] == i:
            for j in range(i, n + 1, i):
                phi[j] -= phi[j] // i
    return phi


def count_consecutive(n: int) -> int:
    if n < 5:
        return 0
    return (n - 4) * (n - 3) // 2


def count_geometric(n: int, phi: List[int]) -> int:
    max_p = iroot4(n)
    total = 0
    for p in range(2, max_p + 1):
        ph = phi[p]
        if ph == 0:
            continue
        p_pow = p * p * p * p
        while p_pow <= n:
            total += ph * (n // p_pow)
            p_pow *= p
    return total


def seq_len_limit(a0: int, a1: int, m: int, s: int, n: int) -> int:
    length = 2
    prev, cur = a0, a1
    while True:
        nxt = m * cur + s * prev
        if nxt > n:
            break
        prev, cur = cur, nxt
        length += 1
    return length


def count_seq_from_recurrence(a0: int, a1: int, m: int, s: int, n: int) -> int:
    length = seq_len_limit(a0, a1, m, s, n)
    if length < 5:
        return 0
    return (length - 4) * (length - 3) // 2


def count_regular(n: int) -> int:
    total = 0
    # s = +1, m = 1 sequence starts at (1,2)
    total += count_seq_from_recurrence(1, 2, 1, 1, n)

    max_m = iroot4(n) + 2
    # s = +1, m >= 2 sequences start at (1,m)
    for m in range(2, max_m + 1):
        length = seq_len_limit(1, m, m, 1, n)
        if length >= 5:
            total += (length - 4) * (length - 3) // 2

    # s = +1, C=2 only for m=2, start (1,3)
    total += count_seq_from_recurrence(1, 3, 2, 1, n)

    # s = -1, m >= 3 sequences start at (1,m)
    for m in range(3, max_m + 1):
        length = seq_len_limit(1, m, m, -1, n)
        if length >= 5:
            total += (length - 4) * (length - 3) // 2

    # s = -1, extra sequences for m=3 and m=4
    total += count_seq_from_recurrence(1, 2, 3, -1, n)
    total += count_seq_from_recurrence(1, 3, 4, -1, n)
    return total


def is_consecutive(seq: List[int]) -> bool:
    return all(seq[i + 1] == seq[i] + 1 for i in range(len(seq) - 1))


def is_geometric(seq: List[int]) -> bool:
    return all(seq[i + 1] * seq[0] == seq[i] * seq[1] for i in range(1, len(seq) - 1))


def is_rec(seq: List[int], m: int, s: int) -> bool:
    return all(
        seq[i + 1] == m * seq[i] + s * seq[i - 1] for i in range(1, len(seq) - 1)
    )


def in_families(seq: List[int]) -> bool:
    if len(seq) < 3:
        return False
    if is_consecutive(seq) or is_geometric(seq):
        return True
    a0, a1, a2 = seq[0], seq[1], seq[2]
    if (a2 - a0) % a1 == 0:
        m = (a2 - a0) // a1
        if m >= 1 and is_rec(seq, m, 1):
            k = a1 * a1 - a0 * (m * a1 + a0)
            if abs(k) <= 2:
                return True
    if (a2 + a0) % a1 == 0:
        m = (a2 + a0) // a1
        if m >= 2 and is_rec(seq, m, -1):
            k = a1 * a1 - a0 * (m * a1 - a0)
            if abs(k) <= 2:
                return True
    return False


def compute_finite_exception_maxes(bound: int) -> List[int]:
    starts = [(1, 2), (2, 3)]
    found: List[List[int]] = []
    for start in starts:
        stack = [list(start)]
        while stack:
            seq = stack.pop()
            if seq[-1] > bound:
                continue
            if len(seq) >= 5 and not in_families(seq):
                found.append(seq)
            a, b = seq[-2], seq[-1]
            t = b * b
            for k in (-2, -1, 0, 1, 2):
                num = t + k
                if num % a == 0:
                    c = num // a
                    if c > b and c <= bound and -2 <= t - a * c <= 2:
                        stack.append(seq + [c])

    # remove prefixes of the infinite exception path 1,2,6,18,54,...
    inf = [1, 2, 6]
    while True:
        nxt = inf[-1] * 3
        if nxt > bound:
            break
        inf.append(nxt)
    inf_prefixes = {tuple(inf[:i]) for i in range(5, len(inf) + 1)}
    maxes = [seq[-1] for seq in found if tuple(seq) not in inf_prefixes]
    maxes.sort()
    return maxes


def infinite_exception_prefix_count(n: int) -> int:
    if n < 1:
        return 0
    length = 1  # term 1
    if n >= 2:
        val = 2
        while val <= n:
            length += 1
            val *= 3
    return max(0, length - 4)


def count_exceptions(n: int, finite_maxes: List[int]) -> int:
    return bisect_right(finite_maxes, n) + infinite_exception_prefix_count(n)


def G(n: int, phi: List[int], finite_maxes: List[int]) -> int:
    total = (
        count_consecutive(n)
        + count_geometric(n, phi)
        + count_regular(n)
        + count_exceptions(n, finite_maxes)
    )
    return total % MOD


def main() -> None:
    max_n = 10**18
    phi = phi_sieve(iroot4(max_n) + 2)
    finite_maxes = compute_finite_exception_maxes(1000)

    assert G(6, phi, finite_maxes) == 4
    assert G(10, phi, finite_maxes) == 26
    assert G(100, phi, finite_maxes) == 4710
    assert G(1000, phi, finite_maxes) == 496805

    print(G(max_n, phi, finite_maxes))


if __name__ == "__main__":
    main()
