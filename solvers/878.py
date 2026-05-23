#!/usr/bin/env python
from __future__ import annotations

"""
Project Euler 878: XOR-Equation B

Carryless multiplication is polynomial multiplication over GF(2).  The
quadratic form is invariant under the orbit step

    (a, b) -> (b, a ^ (b << 1)).

The solver scans only canonical orbit seeds, then walks each seed forward until
the second component exceeds N.
"""


def clmul(x: int, y: int) -> int:
    """Carryless product of two nonnegative integers."""
    if x == 0 or y == 0:
        return 0
    if x.bit_count() < y.bit_count():
        x, y = y, x

    result = 0
    while y:
        bit = y & -y
        result ^= x << (bit.bit_length() - 1)
        y ^= bit
    return result


def value_k(a: int, b: int) -> int:
    return clmul(a, a) ^ (clmul(a, b) << 1) ^ clmul(b, b)


def count_forward_orbit(a: int, b: int, limit: int) -> int:
    total = 0
    while b <= limit:
        total += 1
        a, b = b, a ^ (b << 1)
    return total


def G(limit: int, max_value: int) -> int:
    if max_value == 0:
        return 1

    seed_limit = 1 << ((max_value.bit_length() + 2) // 2)
    squares = [clmul(x, x) for x in range(seed_limit)]

    total = 1  # the fixed orbit (0, 0)
    for a in range(seed_limit):
        square_a = squares[a]
        for b in range(max(1, a), seed_limit):
            invariant = square_a ^ squares[b] ^ (clmul(a, b) << 1)
            if invariant > max_value:
                continue

            # If the predecessor is still ordered, this pair is not the first
            # ordered point on its orbit.
            if (b ^ (a << 1)) > a:
                total += count_forward_orbit(a, b, limit)

    return total


def _self_tests() -> None:
    assert clmul(7, 3) == 9
    assert value_k(3, 6) == 5
    assert G(5, 1) == 4
    assert G(1000, 100) == 398


def main() -> None:
    _self_tests()
    print(G(10**17, 1_000_000))


if __name__ == "__main__":
    main()
