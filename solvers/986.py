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

PREDICT_START_N = 33
SEARCH_WINDOW = 4096


def extinct_for_k1(n: int, k: int) -> bool:
    """
    Decide whether H(1, n) with initial h = k halts (eventual constant 0).
    Uses a dedicated in-place cyclic kernel for c=1, d=n.
    """
    if k == 0:
        return True

    size = n + 1
    last = size - 1
    cells: List[int] = [0] * size
    cells[last] = k
    zero_count = last

    while True:
        for i in range(last):
            old = cells[i]
            nxt = (old + cells[i + 1]) >> 1
            cells[i] = nxt
            if old:
                if not nxt:
                    zero_count += 1
            elif nxt:
                zero_count -= 1

        old = cells[last]
        nxt = (old + cells[0]) >> 1
        cells[last] = nxt
        if old:
            if not nxt:
                zero_count += 1
        elif nxt:
            zero_count -= 1

        # All-zero and all-positive states are forward-invariant.
        if zero_count == size:
            return True
        if zero_count == 0:
            return False


def threshold_k1_plain(n: int) -> int:
    lo = 0
    hi = 1
    while extinct_for_k1(n, hi):
        lo = hi
        hi *= 2

    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if extinct_for_k1(n, mid):
            lo = mid
        else:
            hi = mid
    return lo


def predict_k1_from_previous(s: List[int], n: int) -> int:
    """
    Residue-class cubic extrapolation on n mod 8 from the previous four points.
    """
    a = s[n - 32]
    b = s[n - 24]
    c = s[n - 16]
    d = s[n - 8]
    return d + (d - c) + (d - 2 * c + b) + (d - 3 * c + 3 * b - a)


def threshold_k1_with_guess(n: int, guess: int) -> int:
    lo = max(0, guess - SEARCH_WINDOW)
    hi = guess + SEARCH_WINDOW

    while lo > 0 and not extinct_for_k1(n, lo):
        hi = lo
        lo //= 2

    while extinct_for_k1(n, hi):
        lo = hi
        hi *= 2

    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if extinct_for_k1(n, mid):
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
        if n < PREDICT_START_N:
            s[n] = threshold_k1_plain(n)
        else:
            guess = predict_k1_from_previous(s, n)
            s[n] = threshold_k1_with_guess(n, guess)
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
