#!/usr/bin/env python
"""
Project Euler 505: Bidirectional Recurrence

The boundary of y_n is not at a uniform depth.  Split the frontier of the
complete tree into maximal uniform blocks: blocks wholly before the boundary
evaluate as x-subtrees, while blocks past the boundary evaluate through the
complement transform.  The block values are then folded upward by alternating
min/max operations.
"""

MASK = (1 << 60) - 1


def combine(a: int, b: int, c: int, d: int) -> int:
    return (a * b + c * d) & MASK


def state_at(k: int) -> tuple[int, int]:
    """Return (x(k), x(k//2))."""
    if k == 0:
        return 0, 0

    x = 1
    parent = 0
    for bit in range(k.bit_length() - 2, -1, -1):
        if (k >> bit) & 1:
            next_x = combine(2, x, 3, parent)
        else:
            next_x = combine(3, x, 2, parent)
        parent = x
        x = next_x

    return x, parent


def minimax_subtree(x: int, parent: int, depth: int, alpha: int, beta: int) -> int:
    if depth == 0:
        return x

    left = combine(3, x, 2, parent)
    right = combine(2, x, 3, parent)

    if depth & 1:
        if left < right:
            left, right = right, left
        if depth == 1:
            return left

        value = minimax_subtree(left, x, depth - 1, alpha, beta)
        if value > alpha:
            alpha = value
        if alpha >= beta:
            return alpha

        value = minimax_subtree(right, x, depth - 1, alpha, beta)
        return value if value > alpha else alpha

    if left > right:
        left, right = right, left
    if depth == 1:
        return left

    value = minimax_subtree(left, x, depth - 1, alpha, beta)
    if value < beta:
        beta = value
    if alpha >= beta:
        return beta

    value = minimax_subtree(right, x, depth - 1, alpha, beta)
    return value if value < beta else beta


def evaluate_subtree(k: int, depth: int) -> int:
    x, parent = state_at(k)
    return minimax_subtree(x, parent, depth, 0, MASK)


def collect_blocks(
    start: int, depth: int, left_length: int, blocks: list[tuple[int, int, bool]]
) -> None:
    size = 1 << depth
    if start + size <= left_length:
        blocks.append((start, depth, False))
        return
    if start >= left_length:
        blocks.append((start, depth, True))
        return

    half = size >> 1
    collect_blocks(start, depth - 1, left_length, blocks)
    collect_blocks(start + half, depth - 1, left_length, blocks)


def block_value(start: int, depth: int, right_side: bool, base: int) -> int:
    leaf_start = base + start

    if not right_side:
        return evaluate_subtree(leaf_start >> depth, depth)

    if depth == 0:
        x, _ = state_at(leaf_start >> 1)
        return MASK - x

    return MASK - evaluate_subtree(leaf_start >> depth, depth - 1)


def A(n: int) -> int:
    if n <= 0:
        raise ValueError("n must be positive")
    if n == 1:
        return 1

    total_nodes = 2 * n - 1
    height = total_nodes.bit_length() - 1
    base = 1 << height
    boundary = 2 * n
    left_length = boundary - base

    blocks: list[tuple[int, int, bool]] = []
    collect_blocks(0, height, left_length, blocks)

    values: dict[int, int] = {}
    for start, depth, right_side in blocks:
        values[(start << 6) | depth] = block_value(start, depth, right_side, base)

    def fold(start: int, depth: int) -> int:
        key = (start << 6) | depth
        cached = values.get(key)
        if cached is not None:
            return cached

        half = 1 << (depth - 1)
        left = fold(start, depth - 1)
        right = fold(start + half, depth - 1)
        return max(left, right) if depth & 1 else min(left, right)

    value = fold(0, height)
    return MASK - value if height & 1 else value


def brute_A(n: int) -> int:
    x = [0] * (2 * n + 1)
    if 2 * n >= 1:
        x[1] = 1
    for k in range(1, n):
        current = x[k]
        parent = x[k // 2]
        x[2 * k] = (3 * current + 2 * parent) & MASK
        x[2 * k + 1] = (2 * current + 3 * parent) & MASK

    y = [0] * (2 * n)
    for k in range(2 * n - 1, 0, -1):
        if k >= n:
            y[k] = x[k]
        else:
            y[k] = MASK - max(y[2 * k], y[2 * k + 1])
    return y[1]


def main() -> None:
    assert A(4) == 8
    assert A(10) == (1 << 60) - 34
    assert A(1000) == 101881
    print(A(10**12))


if __name__ == "__main__":
    main()
