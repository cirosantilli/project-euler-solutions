#!/usr/bin/env python
"""
Project Euler 846: Magic Bracelets

Admissible labels are represented by primitive Gaussian-integer vectors.  The
compatibility condition becomes a determinant-one condition, which gives each
label at most two smaller parents.  Sweeping labels downward and carrying open
path fragments over edges counts each bracelet at its unique maximum label.
"""

from math import isqrt


def primes_upto(limit: int) -> list[int]:
    if limit < 2:
        return []

    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, isqrt(limit) + 1):
        if sieve[p]:
            start = p * p
            sieve[start : limit + 1 : p] = b"\x00" * (((limit - start) // p) + 1)

    return [p for p in range(2, limit + 1) if sieve[p]]


def canonical_vector(x: int, y: int) -> tuple[int, int]:
    x = abs(x)
    y = abs(y)
    if y > x:
        x, y = y, x
    return x, y


def build_vectors(limit: int) -> dict[int, tuple[int, int]]:
    square_root = {i * i: i for i in range(isqrt(limit) + 2)}
    vectors: dict[int, tuple[int, int]] = {1: (1, 0), 2: (1, 1)}

    for p in primes_upto(limit):
        if p % 4 != 1:
            continue

        base_x = base_y = 0
        for x in range(1, isqrt(p) + 1):
            y = square_root.get(p - x * x)
            if y is not None:
                base_x, base_y = canonical_vector(x, y)
                break

        if base_x == 0:
            continue

        value = p
        x = base_x
        y = base_y

        while value <= limit:
            cx, cy = canonical_vector(x, y)
            vectors[value] = (cx, cy)

            doubled = 2 * value
            if doubled <= limit:
                vectors[doubled] = canonical_vector(cx - cy, cx + cy)

            x, y = x * base_x - y * base_y, x * base_y + y * base_x
            value *= p

    return vectors


def parent_labels(label: int, vector: tuple[int, int], labels: set[int]) -> tuple[int, ...]:
    if label == 2:
        return (1,)

    x, y = vector
    if y == 0:
        return ()

    beta = pow(y, -1, x)
    alpha = (1 - beta * y) // x

    q1x = abs(beta)
    q1y = abs(alpha)
    q2x = x - q1x
    q2y = y - q1y

    n1 = q1x * q1x + q1y * q1y
    n2 = q2x * q2x + q2y * q2y

    parents = []
    if n1 in labels:
        parents.append(n1)
    if n2 in labels and n2 != n1:
        parents.append(n2)

    parents.sort(reverse=True)
    return tuple(parents)


def edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a >= b else (b, a)


def F(limit: int) -> int:
    vectors = build_vectors(limit)
    labels = set(vectors)
    parents = {
        label: parent_labels(label, vectors[label], labels)
        for label in labels
        if label > 1
    }

    # edge -> (number of open path fragments, sum of interior label totals)
    edge_states: dict[tuple[int, int], tuple[int, int]] = {}
    total = 0

    for label in sorted(labels, reverse=True):
        ps = parents.get(label, ())
        if not ps:
            continue

        for parent in ps:
            key = edge_key(label, parent)
            count, interior_sum = edge_states.get(key, (0, 0))

            # Close older fragments with the real edge label-parent.
            total += count * (label + parent) + interior_sum

            # Seed the trivial open path on this edge.
            edge_states[key] = (count + 1, interior_sum)

        if len(ps) == 2:
            a, b = ps
            count_a, sum_a = edge_states[edge_key(label, a)]
            count_b, sum_b = edge_states[edge_key(label, b)]

            merged_count = count_a * count_b
            merged_sum = merged_count * label + count_a * sum_b + count_b * sum_a

            lower_key = edge_key(a, b)
            old_count, old_sum = edge_states.get(lower_key, (0, 0))
            edge_states[lower_key] = (
                old_count + merged_count,
                old_sum + merged_sum,
            )

    return total


def main() -> None:
    assert F(20) == 258
    assert F(100) == 538768
    print(F(10**6))


if __name__ == "__main__":
    main()
