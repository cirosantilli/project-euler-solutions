#!/usr/bin/env python3
"""Project Euler 847 - Jack's Bean

Compute H(R_19) modulo 1_000_000_007.

No external libraries are used.
"""

from __future__ import annotations

from functools import lru_cache

MOD = 1_000_000_007
INV2 = (MOD + 1) // 2
INV6 = pow(6, MOD - 2, MOD)


def ceil_log2(n: int) -> int:
    """ceil(log2(n)) for n>=1, and 0 for n<=1."""
    if n <= 1:
        return 0
    return (n - 1).bit_length()


def repunit(n: int) -> int:
    """Return R_n = 111...1 (n digits)."""
    r = 0
    for _ in range(n):
        r = r * 10 + 1
    return r


# ---- Small helper sums (mod MOD) ----


def _sum_1(n: int) -> int:
    """Sum_{i=1..n} i (mod MOD)."""
    if n <= 0:
        return 0
    n_mod = n % MOD
    return (n_mod * ((n + 1) % MOD) % MOD) * INV2 % MOD


def _sum_2(n: int) -> int:
    """Sum_{i=1..n} i^2 (mod MOD)."""
    if n <= 0:
        return 0
    n_mod = n % MOD
    a = n_mod * ((n + 1) % MOD) % MOD
    b = (2 * n + 1) % MOD
    return a * b % MOD * INV6 % MOD


def _range_sum_1(l: int, r: int) -> int:
    if l > r:
        return 0
    return (_sum_1(r) - _sum_1(l - 1)) % MOD


def _range_sum_2(l: int, r: int) -> int:
    if l > r:
        return 0
    return (_sum_2(r) - _sum_2(l - 1)) % MOD


def _sum_triples_count(l: int, r: int) -> int:
    """Sum_{s=l..r} C(s+2,2) (mod MOD)."""
    if l > r:
        return 0
    cnt = (r - l + 1) % MOD
    s1 = _range_sum_1(l, r)
    s2 = _range_sum_2(l, r)
    # C(s+2,2) = (s^2 + 3s + 2)/2
    total = (s2 + 3 * s1 + 2 * cnt) % MOD
    return total * INV2 % MOD


# ---- Counting the "bad" triples for a fixed total sum s ----
# A triple (a,b,c) with a+b+c=s is "bad" if it requires one extra question,
# i.e. h(a,b,c) = ceil(log2(s)) + 1.


def _threshold_t(k: int) -> int:
    """t_k = 3*2^(k-2)+2 for k>=3; otherwise effectively +infinity."""
    if k < 3:
        return 10**30
    return 3 * (1 << (k - 2)) + 2


def _base_bad_for_block(M: int, L: int) -> int:
    """Base contribution for s = M + L where M is a power of two and 1<=L<=M.

    Counts triples with sum s where max(a,b,c) < L.
    For our special parameters (cap = L-1), this count simplifies to a plain
    stars-and-bars count once it becomes nonzero.
    """
    # Nonzero iff M+L <= 3*(L-1)  <=>  2L - M - 3 >= 0  <=>  L >= M/2 + 2
    if L < (M >> 1) + 2:
        return 0
    # Let D = 2L - M - 3 >= 0. Then count = C(D+2,2) = C(2L - M - 1, 2).
    n = 2 * L - M - 1
    return n * (n - 1) // 2


@lru_cache(maxsize=None)
def bad_count_for_sum(s: int) -> int:
    """Number of bad triples (a,b,c) with a+b+c=s (exact integer)."""
    if s <= 7:
        return 0
    k = (s - 1).bit_length() - 1  # s in (2^k, 2^(k+1)]
    M = 1 << k
    L = s - M

    res = _base_bad_for_block(M, L)

    # Recursive lift: for large enough L, additional bad triples come from
    # lifting bad triples of sum L by adding M beans to one plate.
    if k >= 3 and L >= _threshold_t(k):
        res += 3 * bad_count_for_sum(L)
    return res


def _sum_base_bad_block_mod(M: int, R: int) -> int:
    """Sum_{L=1..R} base_bad(M,L) (mod MOD) for s=M+L."""
    if R <= 0:
        return 0
    start = (M >> 1) + 2
    if R < start:
        return 0
    a = start
    b = R
    cnt = (b - a + 1) % MOD

    # base(L) = (2L - M - 1)(2L - M - 2)/2
    #         = 2L^2 - (2M+3)L + (M+1)(M+2)/2
    sumL = _range_sum_1(a, b)
    sumL2 = _range_sum_2(a, b)

    const = ((M + 1) % MOD) * ((M + 2) % MOD) % MOD
    const = const * INV2 % MOD

    res = (2 * sumL2 - ((2 * M + 3) % MOD) * sumL + const * cnt) % MOD
    return res


@lru_cache(maxsize=None)
def bad_prefix_sum_mod(N: int) -> int:
    """B(N) = sum_{s=1..N} bad_count_for_sum(s) (mod MOD)."""
    if N <= 0:
        return 0
    if N <= 16:
        # tiny base case by direct summation (cheap and avoids corner bugs)
        return sum(bad_count_for_sum(s) for s in range(1, N + 1)) % MOD

    # Split at the largest power of two <= N.
    pow2 = 1 << (N.bit_length() - 1)
    if N == pow2:
        if pow2 == 1:
            return 0
        M = pow2 >> 1
        return (bad_prefix_sum_mod(M) + _block_bad_sum_mod(M, M)) % MOD

    return (bad_prefix_sum_mod(pow2) + _block_bad_sum_mod(pow2, N - pow2)) % MOD


def _block_bad_sum_mod(M: int, R: int) -> int:
    """Sum_{s=M+1..M+R} bad(s) (mod MOD), where M is a power of two."""
    if R <= 0:
        return 0
    k = M.bit_length() - 1

    res = _sum_base_bad_block_mod(M, R)

    T = _threshold_t(k)
    if T <= R:
        res = (res + 3 * (bad_prefix_sum_mod(R) - bad_prefix_sum_mod(T - 1))) % MOD
    return res


# ---- Base part: sum_{s<=N} C(s+2,2)*ceil_log2(s) ----


def base_part_mod(N: int) -> int:
    if N <= 0:
        return 0

    res = 0
    # m = ceil_log2(s) is constant on s in (2^(m-1), 2^m]
    m = 1
    low = 2  # 2^(m-1)+1 when m=1
    while low <= N:
        high = min((1 << m), N)
        res = (res + m * _sum_triples_count(low, high)) % MOD
        m += 1
        low = (1 << (m - 1)) + 1
    return res


def H_mod(N: int) -> int:
    return (base_part_mod(N) + bad_prefix_sum_mod(N)) % MOD


# ---- Brute-force h(a,b,c) for tiny asserts ----


def _single_plate_questions(n: int) -> int:
    return ceil_log2(n)


@lru_cache(maxsize=None)
def h_bruteforce(a: int, b: int, c: int) -> int:
    """Exact minimal questions for small triples (used only for asserts)."""
    a, b, c = sorted((a, b, c), reverse=True)
    s = a + b + c
    if s <= 1:
        return 0
    if b == 0 and c == 0:
        return _single_plate_questions(a)

    best = 10**9
    plates = (a, b, c)
    for idx, x in enumerate(plates):
        if x == 0:
            continue
        others = [plates[i] for i in range(3) if i != idx]
        y, z = others

        # Ask about k beans on this plate.
        for k in range(1, x):
            cost = 1 + max(_single_plate_questions(k), h_bruteforce(x - k, y, z))
            if cost < best:
                best = cost
        # k == x (ask whole plate)
        cost = 1 + max(_single_plate_questions(x), h_bruteforce(0, y, z))
        if cost < best:
            best = cost

    return best


def H_exact_small(N: int) -> int:
    """Exact H(N) for small N using the derived counting formula."""
    total = 0
    for s in range(1, N + 1):
        total += ((s + 2) * (s + 1) // 2) * ceil_log2(s)
        total += bad_count_for_sum(s)
    return total


def main() -> None:
    # Asserts from the problem statement.
    assert h_bruteforce(1, 2, 3) == 3
    assert h_bruteforce(2, 3, 3) == 4
    assert H_exact_small(6) == 203
    assert H_exact_small(20) == 7718
    assert H_exact_small(repunit(3)) == 1634144

    n = repunit(19)
    print(H_mod(n) % MOD)


if __name__ == "__main__":
    main()
