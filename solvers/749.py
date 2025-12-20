#!/usr/bin/env python3
"""
Project Euler 749: Near Power Sums

A positive integer n is a near power sum if there exists a positive integer k such that
sum(digit^k) over the decimal digits of n equals n-1 or n+1.

This program computes S(16), the sum of all near power sum numbers with at most 16 digits.

No external libraries are used.
"""

from __future__ import annotations


def _build_pow10(max_digits: int) -> list[int]:
    p10 = [1] * (max_digits + 1)
    for i in range(1, max_digits + 1):
        p10[i] = p10[i - 1] * 10
    return p10


def _max_k_for_digits(max_digits: int, p10: list[int]) -> int:
    # Safe global bound: if there is any digit >= 2, then 2^k <= n+1 <= 10^max_digits.
    # Find smallest K with 2^K > 10^max_digits, then the needed max k is at most K.
    limit = p10[max_digits]
    k = 0
    p = 1
    while p <= limit:
        k += 1
        p *= 2
    return k


def _build_pow_table(max_k: int) -> list[list[int]]:
    # pow_table[k][d] = d^k for k>=1, d in 0..9
    pow_table = [[0] * 10 for _ in range(max_k + 1)]
    for d in range(10):
        pow_table[1][d] = d
    for k in range(2, max_k + 1):
        row = pow_table[k]
        prev = pow_table[k - 1]
        for d in range(10):
            row[d] = prev[d] * d
    return pow_table


def _build_k_bounds(
    max_digits: int, max_k: int, p10: list[int]
) -> tuple[list[list[int]], list[list[list[int]]]]:
    """
    For length L and max digit m (>=2):

      t(k) = sum(c_d * d^k)

    We use coarse bounds:
      c_m * m^k <= t(k) <= L * m^k

    Since n is L digits, n ~ t(k), so we only need k where t(k) could have the right magnitude:
      10^(L-1)-1 <= t(k) <= 10^L

    Precompute:
      k_low[L][m]  = smallest k with L*m^k >= 10^(L-1)-1  (otherwise t(k) is too small)
      k_high[L][m][c] = largest k with c*m^k <= 10^L      (otherwise t(k) is too large)
    """
    low_t = [0] * (max_digits + 1)
    high_t = [0] * (max_digits + 1)
    for L in range(1, max_digits + 1):
        low_t[L] = p10[L - 1] - 1
        high_t[L] = p10[L]

    k_low = [[max_k + 1] * 10 for _ in range(max_digits + 1)]
    k_high = [
        [[0] * (max_digits + 1) for _ in range(10)] for __ in range(max_digits + 1)
    ]

    for L in range(1, max_digits + 1):
        lt = low_t[L]
        ht = high_t[L]
        for m in range(2, 10):
            # k_low depends only on (L,m)
            p = m
            k = 1
            while k <= max_k and L * p < lt:
                p *= m
                k += 1
            k_low[L][m] = k

            # k_high depends on (L,m,c_m)
            for c in range(1, L + 1):
                p = m
                best = 0
                for kk in range(1, max_k + 1):
                    if c * p <= ht:
                        best = kk
                        p *= m
                    else:
                        break
                k_high[L][m][c] = best

    return k_low, k_high


def _build_digit_pack_tables() -> tuple[list[int], list[list[int]]]:
    """
    We represent a digit multiset as a packed integer with 5 bits per digit:
      code = sum( count[d] << (5*d) ), and counts are <= 16 so 5 bits are enough.

    To compute the packed digit counts of an integer quickly, we precompute a table for 4-digit chunks.
    """
    pack4 = [0] * 10000
    for x in range(10000):
        y = x
        code = 0
        # exactly 4 digits (including leading zeros)
        d = y % 10
        code += 1 << (5 * d)
        y //= 10
        d = y % 10
        code += 1 << (5 * d)
        y //= 10
        d = y % 10
        code += 1 << (5 * d)
        y //= 10
        d = y % 10
        code += 1 << (5 * d)
        pack4[x] = code

    # pack_exact[len][x] = packed digit counts for x written in exactly `len` digits (no extra leading zeros).
    # For len=4 this is the same as pack4.
    pack_exact = [[], [0] * 10, [0] * 100, [0] * 1000, pack4]

    for x in range(10):
        pack_exact[1][x] = 1 << (5 * x)

    for x in range(100):
        y = x
        code = 0
        d = y % 10
        code += 1 << (5 * d)
        y //= 10
        d = y % 10
        code += 1 << (5 * d)
        pack_exact[2][x] = code

    for x in range(1000):
        y = x
        code = 0
        d = y % 10
        code += 1 << (5 * d)
        y //= 10
        d = y % 10
        code += 1 << (5 * d)
        y //= 10
        d = y % 10
        code += 1 << (5 * d)
        pack_exact[3][x] = code

    return pack4, pack_exact


def _pack_digits_len(
    n: int, L: int, pack4: list[int], pack_exact: list[list[int]]
) -> int:
    """
    Packed digit counts of n, assuming n has exactly L digits.
    Splits n into 4-digit chunks (lower chunks are always 4 digits including leading zeros).
    """
    if L <= 4:
        return pack_exact[L][n]
    if L <= 8:
        a = n % 10000
        b = n // 10000
        return pack4[a] + pack_exact[L - 4][b]
    if L <= 12:
        a = n % 10000
        n //= 10000
        b = n % 10000
        c = n // 10000
        return pack4[a] + pack4[b] + pack_exact[L - 8][c]
    # L <= 16
    a = n % 10000
    n //= 10000
    b = n % 10000
    n //= 10000
    c = n % 10000
    d = n // 10000
    return pack4[a] + pack4[b] + pack4[c] + pack_exact[L - 12][d]


def near_power_sums_by_length(max_digits: int) -> list[set[int]]:
    """
    Returns a list results_by_len where results_by_len[L] is the set of near power sums with exactly L digits.
    """
    p10 = _build_pow10(max_digits)
    max_k = _max_k_for_digits(max_digits, p10)
    pow_table = _build_pow_table(max_k)
    k_low, k_high = _build_k_bounds(max_digits, max_k, p10)
    pack4, pack_exact = _build_digit_pack_tables()

    results_by_len: list[set[int]] = [set() for _ in range(max_digits + 1)]

    for L in range(1, max_digits + 1):
        lo = p10[L - 1]
        hi = p10[L]

        for m in range(2, 10):
            shift_m = 5 * m
            km_base = k_low[L][m]

            for c_m in range(1, L + 1):
                k1 = km_base
                k2 = k_high[L][m][c_m]
                if k1 > k2:
                    continue

                Ks = list(range(k1, k2 + 1))
                rows = [pow_table[k] for k in Ks]
                base_t = [c_m * pow_table[k][m] for k in Ks]
                rem = L - c_m

                counts = [0] * m  # digits 0..m-1
                sig_m = c_m << shift_m

                def leaf(packed_part: int) -> None:
                    sig = packed_part | sig_m
                    for idx, row in enumerate(rows):
                        t = base_t[idx]
                        for d in range(m):
                            c = counts[d]
                            if c:
                                t += c * row[d]

                        n = t - 1
                        if (
                            lo <= n < hi
                            and _pack_digits_len(n, L, pack4, pack_exact) == sig
                        ):
                            results_by_len[L].add(n)

                        n = t + 1
                        if (
                            lo <= n < hi
                            and _pack_digits_len(n, L, pack4, pack_exact) == sig
                        ):
                            results_by_len[L].add(n)

                if m == 2:
                    # counts[0]=c0, counts[1]=rem-c0
                    for c0 in range(rem + 1):
                        c1 = rem - c0
                        counts[0] = c0
                        counts[1] = c1
                        leaf((c0 << 0) | (c1 << 5))
                else:
                    # Recursive stars-and-bars over digits 0..m-1
                    def rec(d: int, remaining: int, packed_part: int) -> None:
                        if d == m - 1:
                            counts[d] = remaining
                            leaf(packed_part | (remaining << (5 * d)))
                            return
                        shift = 5 * d
                        for c in range(remaining + 1):
                            counts[d] = c
                            rec(d + 1, remaining - c, packed_part | (c << shift))

                    rec(0, rem, 0)

    return results_by_len


def S_from_results(results_by_len: list[set[int]], d: int) -> int:
    return sum(
        n
        for L in range(1, min(d, len(results_by_len) - 1) + 1)
        for n in results_by_len[L]
    )


def main() -> None:
    results = near_power_sums_by_length(16)

    # Test values from the problem statement.
    assert S_from_results(results, 2) == 110
    assert S_from_results(results, 6) == 2562701

    print(S_from_results(results, 16))


if __name__ == "__main__":
    main()
