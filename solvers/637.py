#!/usr/bin/env python3
"""
Project Euler 637 - Flexible Digit Sum

We compute g(n, 10, 3) by scanning all i <= n, evaluating f(i, 10) and f(i, 3).
For each base B, f(i, B) can only be 0..3 in our range because splitting all
digits once gives a small sum and then at most two more steps are needed.

For i >= B with digit sum >= B, f(i, B) = 2 iff there exists a partition of the
digits whose block-sum has digit sum < B; otherwise f(i, B) = 3. We test this
by enumerating digit partitions in a good order (many separators first), using
prefix values and precomputed digit sums for fast checking.
"""

from typing import List, Tuple


def _digit_sum_table(limit: int, base: int) -> bytearray:
    table = bytearray(limit + 1)
    for i in range(1, limit + 1):
        table[i] = table[i // base] + i % base
    return table


def _powers(base: int, max_len: int) -> List[int]:
    pw = [1] * (max_len + 1)
    for i in range(max_len):
        pw[i + 1] = pw[i] * base
    return pw


def _blocks_by_length(max_len: int) -> List[List[Tuple[int, ...]]]:
    blocks: List[List[Tuple[int, ...]]] = [None] * (max_len + 1)  # type: ignore[list-item]
    for length in range(2, max_len + 1):
        masks = list(range(1 << (length - 1)))
        masks.sort(key=lambda m: m.bit_count(), reverse=True)
        blocks_len: List[Tuple[int, ...]] = []
        for mask in masks:
            parts: List[int] = []
            start = 0
            for i in range(length - 1):
                if (mask >> (length - 2 - i)) & 1:
                    parts.append(start)
                    parts.append(i - start + 1)
                    start = i + 1
            parts.append(start)
            parts.append(length - start)
            blocks_len.append(tuple(parts))
        blocks[length] = blocks_len
    return blocks


def g(limit: int, base1: int, base2: int) -> int:
    if base1 != 10 or base2 != 3:
        raise ValueError("This solver only supports bases 10 and 3.")

    max_len10 = 8
    max_len3 = 15

    pow10 = _powers(10, max_len10)
    pow3 = _powers(3, max_len3)
    blocks10 = _blocks_by_length(max_len10)
    blocks3 = _blocks_by_length(max_len3)

    ds10 = _digit_sum_table(limit, 10)
    ds3 = _digit_sum_table(limit, 3)

    digits10 = [0] * max_len10  # least significant digit first
    digits3 = [0] * max_len3
    len10 = 1
    len3 = 1
    sum10 = 0
    sum3 = 0
    prefix10 = [0] * (max_len10 + 1)
    prefix3 = [0] * (max_len3 + 1)

    total = 0

    pow10_local = pow10
    pow3_local = pow3
    blocks10_local = blocks10
    blocks3_local = blocks3
    ds10_local = ds10
    ds3_local = ds3

    for n in range(1, limit + 1):
        # Increment base-10 digits and sum.
        i = 0
        while i < len10 and digits10[i] == 9:
            sum10 -= 9
            digits10[i] = 0
            i += 1
        if i == len10:
            digits10[i] = 1
            len10 += 1
            sum10 += 1
        else:
            digits10[i] += 1
            sum10 += 1

        # Increment base-3 digits and sum.
        i = 0
        while i < len3 and digits3[i] == 2:
            sum3 -= 2
            digits3[i] = 0
            i += 1
        if i == len3:
            digits3[i] = 1
            len3 += 1
            sum3 += 1
        else:
            digits3[i] += 1
            sum3 += 1

        # f(n, 10)
        if len10 == 1:
            f10 = 0
        elif sum10 < 10:
            f10 = 1
        else:
            p = 0
            prefix10[0] = 0
            pos = 0
            for idx in range(len10 - 1, -1, -1):  # msd -> lsd
                p = p * 10 + digits10[idx]
                pos += 1
                prefix10[pos] = p
            good = False
            for blk in blocks10_local[len10]:
                s = 0
                for j in range(0, len(blk), 2):
                    start = blk[j]
                    ln = blk[j + 1]
                    s += prefix10[start + ln] - prefix10[start] * pow10_local[ln]
                if ds10_local[s] < 10:
                    good = True
                    break
            f10 = 2 if good else 3

        # f(n, 3)
        if len3 == 1:
            f3 = 0
        elif sum3 < 3:
            f3 = 1
        else:
            p = 0
            prefix3[0] = 0
            pos = 0
            for idx in range(len3 - 1, -1, -1):
                p = p * 3 + digits3[idx]
                pos += 1
                prefix3[pos] = p
            good = False
            for blk in blocks3_local[len3]:
                s = 0
                for j in range(0, len(blk), 2):
                    start = blk[j]
                    ln = blk[j + 1]
                    s += prefix3[start + ln] - prefix3[start] * pow3_local[ln]
                if ds3_local[s] < 3:
                    good = True
                    break
            f3 = 2 if good else 3

        if f10 == f3:
            total += n

    return total


def main() -> None:
    assert g(100, 10, 3) == 3302
    print(g(10_000_000, 10, 3))


if __name__ == "__main__":
    main()
