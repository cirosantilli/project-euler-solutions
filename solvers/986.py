from __future__ import annotations

from math import gcd
from typing import Dict, List, Tuple


LIMIT = 160

# For reduced pairs (c, 1), these are the only exceptions to
# H(c, 1) = S[1 + (c - 1) // 2], where G = 2 * H + 1.
EXCEPTION_H: Dict[int, int] = {
    2: 3,
    3: 5,
    4: 7,
    5: 11,
    6: 13,
    8: 21,
    10: 31,
}


def eventual_constant(c: int, d: int, h0: int) -> int:
    """
    Reverse-process carry recurrence:
        t_n = floor((t_{n-d} + t_{n-(c+d)}) / 2), t_0 = h0, t_n = 0 for n < 0.
    Returns the eventual constant value of t_n (0 means halting).
    """
    w = c + d
    ring: List[int] = [0] * w
    ring[0] = h0
    counts: Dict[int, int] = {h0: 1}

    n = 0
    while True:
        n += 1
        pos = n % w

        a = ring[(n - d) % w] if n >= d else 0
        b = ring[pos] if n >= w else 0
        v = (a + b) // 2

        if n >= w:
            old = ring[pos]
            cnt = counts[old]
            if cnt == 1:
                del counts[old]
            else:
                counts[old] = cnt - 1

        ring[pos] = v
        counts[v] = counts.get(v, 0) + 1

        # Once the last w values are all equal, the sequence is fixed forever.
        if n >= w - 1 and len(counts) == 1:
            return next(iter(counts))


def threshold_h(c: int, d: int) -> int:
    """
    Maximum h such that eventual_constant(c, d, h) == 0.
    """
    lo = 0
    hi = 1
    while eventual_constant(c, d, hi) == 0:
        lo = hi
        hi *= 2

    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if eventual_constant(c, d, mid) == 0:
            lo = mid
        else:
            hi = mid
    return lo


def build_s_sequence(max_n: int) -> List[int]:
    """
    S[n] = H(1, n), where G(1, n) = 2*S[n] + 1.
    """
    s = [0] * (max_n + 1)
    for n in range(1, max_n + 1):
        s[n] = threshold_h(1, n)
    return s


def h_reduced(c: int, d: int, s: List[int]) -> int:
    """
    H(c, d) for reduced (gcd(c,d)=1), using the proven reduction:
      H(c,d) = S[d + (c-1)//2], except a finite set for d=1.
    """
    if d == 1 and c in EXCEPTION_H:
        return EXCEPTION_H[c]
    return s[d + (c - 1) // 2]


def g_value(c: int, d: int, s: List[int]) -> int:
    g = gcd(c, d)
    cr = c // g
    dr = d // g
    h = h_reduced(cr, dr, s)
    return 2 * h + 1


def solve(limit: int = LIMIT) -> int:
    max_n = limit + (limit - 1) // 2
    s = build_s_sequence(max_n)

    # Problem-statement checks.
    assert g_value(2, 1, s) == 7
    assert g_value(1, 2, s) == 7
    assert g_value(3, 1, s) == 11
    assert g_value(2, 2, s) == 3
    assert g_value(1, 3, s) == 15

    memo: Dict[Tuple[int, int], int] = {}
    total = 0
    for c in range(1, limit + 1):
        for d in range(1, limit + 1):
            g = gcd(c, d)
            key = (c // g, d // g)
            val = memo.get(key)
            if val is None:
                h = h_reduced(key[0], key[1], s)
                val = 2 * h + 1
                memo[key] = val
            total += val
    return total


def main() -> None:
    print(solve())


if __name__ == "__main__":
    main()

