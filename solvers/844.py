#!/usr/bin/env python3
"""
Project Euler 844: k-Markov Numbers

We work with positive integer solutions to:
    sum_{i=1..k} x_i^2 = k * prod_{i=1..k} x_i

A k-Markov number is any value appearing as a coordinate in some solution.
Let M_k(N) be the sum of distinct k-Markov numbers <= N.
Let S(K,N) = sum_{k=3..K} M_k(N).

We must compute S(10^18, 10^18) modulo 1_405_695_061.

No external libraries are used (standard library only).
"""

from bisect import bisect_left, insort
from math import isqrt

MOD = 1_405_695_061


def _replace_one(sorted_tuple, old, new):
    """Replace one occurrence of `old` with `new` in a sorted tuple, returning a new sorted tuple."""
    lst = list(sorted_tuple)
    i = bisect_left(lst, old)
    # old must exist
    assert i < len(lst) and lst[i] == old
    lst.pop(i)
    insort(lst, new)
    return tuple(lst)


def Mk_sum(k: int, N: int, mod: int | None = None) -> int:
    """
    Compute M_k(N): sum of distinct k-Markov numbers <= N.

    Uses Vieta jumping / root-flipping on the Markov–Hurwitz equation.
    We represent a solution state only by its non-1 entries (sorted tuple);
    the remaining entries are implicit 1s.

    If `mod` is provided, the running sum is reduced modulo `mod`.
    """
    if N < 1:
        return 0

    # 1 is always a k-Markov number (solution is all ones)
    seen_numbers = {1}
    s = 1
    if mod is not None:
        s %= mod

    # If k-1 > N then the only k-Markov number <=N is 1
    if k - 1 > N:
        return s if mod is None else (s % mod)

    # DFS over the increasing-move solution tree.
    # State stores only non-1 values; also keep their product for speed.
    start = ()
    stack = [(start, 1)]
    visited_states = {start}

    while stack:
        non_ones, prod_non_ones = stack.pop()
        ones = k - len(non_ones)

        # record coordinates as k-Markov numbers
        for v in non_ones:
            if v <= N and v not in seen_numbers:
                seen_numbers.add(v)
                s += v
                if mod is not None:
                    s %= mod

        # distinct coordinate values we can attempt to "jump"
        values = set(non_ones)
        if ones > 0:
            values.add(1)

        for v in values:
            if v == 1:
                # Jump a 1: x' = k*(product of other coords) - 1
                if ones == 0:
                    continue
                new = k * prod_non_ones - 1
                if new <= 1 or new > N:
                    continue
                # now that 1 becomes a non-one; product multiplies by new
                new_state = tuple(sorted(non_ones + (new,)))
                new_prod = prod_non_ones * new
            else:
                # Jump a non-one: x' = k*(product of other coords) - x
                new = k * (prod_non_ones // v) - v
                if new <= v or new > N:
                    continue
                new_state = _replace_one(non_ones, v, new)
                new_prod = (prod_non_ones // v) * new

            if new_state not in visited_states:
                visited_states.add(new_state)
                stack.append((new_state, new_prod))

    return s if mod is None else (s % mod)


def _m3_min_third(k: int) -> int:
    """
    Minimal possible value of the 3rd non-1 entry obtainable from the all-ones solution,
    via successive 1-jumps:
      a1 = k-1
      a2 = k(k-1)-1
      a3_min = k*(k-1)*a2 - 1

    Algebraically: a3_min = k^4 - 2k^3 + k - 1
    """
    return k * k * k * k - 2 * k * k * k + k - 1


def _a2(k: int) -> int:
    return k * k - k - 1


def _a3(k: int) -> int:
    return k * k * k - k * k - 2 * k + 1


def _max_k_monotone_true(limit: int, start: int, pred) -> int:
    """
    Find max k <= limit with pred(k)=True, assuming pred is monotone (True then False).
    Returns start-1 if pred(start) is False.
    """
    if limit < start:
        return limit
    if not pred(start):
        return start - 1

    lo = start
    hi = start
    # exponential search for an upper bound where pred is False (or hit limit)
    while hi < limit and pred(hi):
        lo = hi
        hi = min(limit, hi * 2)

    if pred(hi):
        return hi

    # binary search in (lo, hi)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if pred(mid):
            lo = mid
        else:
            hi = mid
    return lo


def max_k_three_nonones(N: int, K: int) -> int:
    """Largest k<=K such that a 3-non-one solution can still have max<=N."""
    if K < 3:
        return K
    return _max_k_monotone_true(K, 3, lambda k: _m3_min_third(k) <= N)


def max_k_a2(N: int, K: int) -> int:
    """Largest k<=K with a2(k)=k^2-k-1 <= N."""
    if K < 1:
        return 0
    disc = 1 + 4 * (N + 1)
    r = (1 + isqrt(disc)) // 2
    r = min(r, K)
    while r > 0 and _a2(r) > N:
        r -= 1
    return r


def max_k_a3(N: int, K: int) -> int:
    """Largest k<=K with a3(k)=k^3-k^2-2k+1 <= N."""
    if K < 3:
        return K
    if _a3(3) > N:
        return 2
    return _max_k_monotone_true(K, 3, lambda k: _a3(k) <= N)


def _sum1(n: int) -> int:
    return n * (n + 1) // 2


def _sum2(n: int) -> int:
    return n * (n + 1) * (2 * n + 1) // 6


def _sum3(n: int) -> int:
    t = n * (n + 1) // 2
    return t * t


def _range_sum(func, l: int, r: int) -> int:
    if l > r:
        return 0
    return func(r) - func(l - 1)


def S_mod(K: int, N: int, mod: int = MOD) -> int:
    """
    Compute S(K,N) modulo `mod`.

    Key decomposition for large k:
      If k-1 > N, then M_k(N)=1.

      Let cutoff be the largest k such that a 3-non-one solution can still keep max<=N.
      For all k > cutoff, every solution with max<=N has at most 2 non-1 entries, and
      also the 4th term of the 2-non-one chain exceeds N, so:
        - if a3(k) <= N:  M_k(N) = 1 + a1 + a2 + a3 = k^3 - 2k
        - elif a2(k) <= N: M_k(N) = 1 + a1 + a2 = k^2 - 1
        - else:            M_k(N) = 1 + a1 = k
    """
    if K < 3 or N < 1:
        return 0

    # k > N+1 contributes only {1}
    K_eff = min(K, N + 1)
    total = 0

    cutoff = min(K_eff, max_k_three_nonones(N, K_eff))

    # Enumerate exactly for k <= cutoff
    for k in range(3, cutoff + 1):
        total = (total + Mk_sum(k, N, mod=mod)) % mod

    start = cutoff + 1
    if start > K_eff:
        # add trailing ones if K > N+1
        if K > K_eff:
            total = (total + (K - K_eff)) % mod
        return total

    # Polynomial ranges for k > cutoff
    k3 = min(K_eff, max_k_a3(N, K_eff))
    k2 = min(K_eff, max_k_a2(N, K_eff))

    # Region 1: a3 <= N  => M_k = k^3 - 2k
    l, r = start, k3
    if l <= r:
        part = (_range_sum(_sum3, l, r) - 2 * _range_sum(_sum1, l, r)) % mod
        total = (total + part) % mod

    # Region 2: a2 <= N < a3  => M_k = k^2 - 1
    l, r = max(start, k3 + 1), k2
    if l <= r:
        part = (_range_sum(_sum2, l, r) - (r - l + 1)) % mod
        total = (total + part) % mod

    # Region 3: a2 > N  => M_k = k
    l, r = max(start, k2 + 1), K_eff
    if l <= r:
        part = _range_sum(_sum1, l, r) % mod
        total = (total + part) % mod

    # k > N+1 contributes 1 each
    if K > K_eff:
        total = (total + (K - K_eff)) % mod

    return total % mod


def S_exact_small(K: int, N: int) -> int:
    """Exact S(K,N) for small K, used only for validating given examples."""
    return sum(Mk_sum(k, N) for k in range(3, K + 1))


def _run_asserts() -> None:
    # Test values given in the problem statement:
    assert Mk_sum(3, 10**3) == 2797
    assert Mk_sum(8, 10**8) == 131493335
    assert S_exact_small(4, 10**2) == 229
    assert S_exact_small(10, 10**8) == 2383369980


def main() -> None:
    _run_asserts()
    K = 10**18
    N = 10**18
    print(S_mod(K, N, mod=MOD))


if __name__ == "__main__":
    main()
