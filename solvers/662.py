#!/usr/bin/env python
"""
Project Euler 662: Fibonacci Paths

Alice can step from (a,b) to (a+x, b+y) when:
  - x >= 0, y >= 0
  - sqrt(x^2 + y^2) is a Fibonacci number (1,2,3,5,8,...)

We compute F(W,H) modulo 1_000_000_007.

Notes:
- No external libraries are used.
- Test values from the problem statement are asserted.
"""

from array import array
from math import isqrt

MOD = 1_000_000_007


def fibs_upto(n: int) -> list[int]:
    """Return Fibonacci numbers starting with 1,2 up to n (inclusive)."""
    if n < 1:
        return []
    fibs = [1, 2]
    while fibs[-1] + fibs[-2] <= n:
        fibs.append(fibs[-1] + fibs[-2])
    return fibs


def step_vectors(W: int, H: int) -> list[tuple[int, int]]:
    """
    Enumerate all step vectors (dx,dy) with 0<=dx<=W, 0<=dy<=H, not both 0,
    such that sqrt(dx^2+dy^2) is a Fibonacci number.
    """
    max_len = isqrt(W * W + H * H)
    fibs = fibs_upto(max_len)

    steps = set()
    for f in fibs:
        ff = f * f
        # dx beyond min(f,W) can't work since dx^2 <= f^2
        lim = min(f, W)
        for dx in range(lim + 1):
            dy2 = ff - dx * dx
            dy = isqrt(dy2)
            if dy * dy == dy2 and dy <= H:
                if dx or dy:
                    steps.add((dx, dy))
    return sorted(steps)


def count_paths(W: int, H: int, mod: int = MOD) -> int:
    """
    Dynamic programming over columns.

    First accumulate every contribution from earlier columns, then fill the
    current column bottom-up so vertical same-column dependencies are ready.
    """
    if W < 0 or H < 0:
        return 0
    if W == 0 and H == 0:
        return 1

    steps = step_vectors(W, H)

    vertical = []
    previous_columns = []
    max_dx = 0
    for dx, dy in steps:
        if dx == 0:
            vertical.append(dy)
        else:
            previous_columns.append((dx, dy))
            if dx > max_dx:
                max_dx = dx

    vertical.sort()
    previous_columns.sort()

    buf = max_dx + 1
    buffer = [array("I", [0]) * (H + 1) for _ in range(buf)]

    for x in range(W + 1):
        column = buffer[x % buf]
        incoming = [0] * (H + 1)

        for dx, dy in previous_columns:
            if dx > x:
                break
            source = buffer[(x - dx) % buf]
            if dy == 0:
                for y in range(H + 1):
                    incoming[y] += source[y]
            else:
                limit = H - dy
                for y in range(limit + 1):
                    incoming[y + dy] += source[y]

        if x == 0:
            incoming[0] += 1

        for y in range(H + 1):
            val = incoming[y]
            for dy in vertical:
                if dy > y:
                    break
                val += column[y - dy]
            column[y] = val % mod

    return int(buffer[W % buf][H])


def main() -> None:
    # Given test values
    assert count_paths(3, 4) == 278
    assert count_paths(10, 10) == 215846462

    # Required output (do not hardcode/assert the final answer)
    W = 10_000
    H = 10_000
    print(count_paths(W, H) % MOD)


if __name__ == "__main__":
    main()
