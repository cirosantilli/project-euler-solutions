#!/usr/bin/env python3
"""Project Euler 879: Touch-screen Password

Counts the number of distinct passwords on an n x n grid under the rules:
- A password is the sequence of distinct spots visited (length >= 2).
- A straight segment from A to B automatically includes any *currently visible*
  intermediate grid spots lying on the open segment, in geometric order.
- Once visited, a spot disappears: it cannot be visited again and is ignored by
  future segments that pass through its location.

This program asserts the sample value given in the problem statement (3x3) and
prints the answer for 4x4.

No external libraries are used.
"""

from __future__ import annotations

import math
from typing import List


def _precompute_between_masks(n: int) -> List[List[int]]:
    """between[a][b] is a bitmask of lattice points strictly between a and b."""
    N = n * n
    coords = [(i % n, i // n) for i in range(N)]
    between = [[0] * N for _ in range(N)]

    for a in range(N):
        xa, ya = coords[a]
        row = between[a]
        for b in range(N):
            if a == b:
                continue
            xb, yb = coords[b]
            dx = xb - xa
            dy = yb - ya
            g = math.gcd(abs(dx), abs(dy))
            if g <= 1:
                continue
            sx = dx // g
            sy = dy // g
            mask = 0
            # Points strictly between: k=1..g-1
            for k in range(1, g):
                x = xa + sx * k
                y = ya + sy * k
                mask |= 1 << (y * n + x)
            row[b] = mask

    return between


def count_passwords(n: int) -> int:
    """Return number of valid passwords on an n x n grid."""
    N = n * n
    if N < 2:
        return 0

    between = _precompute_between_masks(n)
    all_mask = (1 << N) - 1

    # dp[cur][mask] = number of non-empty continuations from (cur, visited=mask)
    # where mask includes cur.
    dp = [[0] * (1 << N) for _ in range(N)]

    # Process masks in descending popcount so that (mask | bit) is already known.
    buckets: List[List[int]] = [[] for _ in range(N + 1)]
    for mask in range(1 << N):
        buckets[mask.bit_count()].append(mask)

    dp_rows = dp  # local alias
    for k in range(N, 0, -1):
        for mask in buckets[k]:
            remaining = all_mask ^ mask
            if remaining == 0:
                continue

            m = mask
            while m:
                lsb = m & -m
                cur = lsb.bit_length() - 1
                m ^= lsb

                between_row = between[cur]
                total = 0

                rem = remaining
                while rem:
                    bit = rem & -rem
                    nxt = bit.bit_length() - 1
                    rem ^= bit

                    # nxt is directly selectable iff all intermediate points are already visited.
                    if (between_row[nxt] & remaining) == 0:
                        total += 1 + dp_rows[nxt][mask | bit]

                dp_rows[cur][mask] = total

    total_passwords = 0
    for start in range(N):
        total_passwords += dp_rows[start][1 << start]

    return total_passwords


def _simulate_trace_endpoints(n: int, endpoints_1based: List[int]) -> List[int]:
    """Interpret a traced endpoint sequence, returning the resulting password.

    Used only for validating the examples in the problem statement on a 3x3 grid.
    Spots are numbered 1..n^2 in row-major order.
    """
    assert len(endpoints_1based) >= 2
    N = n * n

    endpoints = [e - 1 for e in endpoints_1based]
    coords = [(i % n, i // n) for i in range(N)]

    visited = 0
    password: List[int] = []

    cur = endpoints[0]
    visited |= 1 << cur
    password.append(cur)

    for dest in endpoints[1:]:
        if (visited >> dest) & 1:
            raise ValueError("Trace uses a spot that has already disappeared.")

        x0, y0 = coords[cur]
        x1, y1 = coords[dest]
        dx = x1 - x0
        dy = y1 - y0
        g = math.gcd(abs(dx), abs(dy))
        sx = dx // g
        sy = dy // g

        # Walk along the segment, including currently visible intermediate points.
        for k in range(1, g + 1):
            x = x0 + sx * k
            y = y0 + sy * k
            idx = y * n + x
            if ((visited >> idx) & 1) == 0:
                visited |= 1 << idx
                password.append(idx)

        cur = dest

    # Convert back to 1-based labels
    return [p + 1 for p in password]


def main() -> None:
    # Examples from the statement (3x3 labels 1..9):
    assert _simulate_trace_endpoints(3, [1, 9]) == [1, 5, 9]
    assert _simulate_trace_endpoints(3, [1, 9, 3, 7]) == [1, 5, 9, 6, 3, 7]

    # Test value from the statement:
    assert count_passwords(3) == 389488

    # Required output:
    print(count_passwords(4))


if __name__ == "__main__":
    main()
