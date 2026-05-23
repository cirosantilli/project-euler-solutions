#!/usr/bin/env python
"""
Project Euler 623 - Lambda Count

Let A_d(s) be the number of exact-size lambda terms in a context with d
available binders.  Variables contribute d choices at size 1, an abstraction
uses one term from depth d + 1 at size s - 5, and an application is an ordered
pair of terms from the same depth whose sizes sum to s - 2.

For each depth we sweep sizes upward and maintain the ordered convolution
    C(k) = sum_{i+j=k} A_d(i) A_d(j)
incrementally.  The depth layers are processed from deepest to shallowest, so
only the current layer, the next deeper layer, and one convolution buffer are
needed.
"""

MOD = 1_000_000_007
LIMIT = 2000


def exact_counts(limit):
    dmax = (limit - 1) // 5
    next_layer = [0] * (limit + 1)

    for depth in range(dmax, -1, -1):
        max_size = limit - 5 * depth
        current = [0] * (max_size + 1)
        convolution = [0] * (max_size + 1)
        max_convolution_index = max_size - 2

        for size in range(1, max_size + 1):
            value = convolution[size - 2] if size >= 2 else 0

            if size == 1:
                value += depth
            if size >= 5:
                shifted = size - 5
                if shifted < len(next_layer):
                    value += next_layer[shifted]

            value %= MOD
            current[size] = value

            if not value:
                continue

            double_value = 2 * value
            upper = min(size - 1, max_convolution_index - size)
            for other_size in range(1, upper + 1):
                index = size + other_size
                convolution[index] = (
                    convolution[index] + double_value * current[other_size]
                ) % MOD

            diagonal = 2 * size
            if diagonal <= max_convolution_index:
                convolution[diagonal] = (
                    convolution[diagonal] + value * value
                ) % MOD

        next_layer = current

    return next_layer


def solve():
    counts = exact_counts(LIMIT)

    cumulative = 0
    checkpoints = {}
    for size, count in enumerate(counts):
        cumulative = (cumulative + count) % MOD
        if size in (6, 9, 15, 35):
            checkpoints[size] = cumulative

    assert checkpoints[6] == 1
    assert checkpoints[9] == 2
    assert checkpoints[15] == 20
    assert checkpoints[35] == 3166438

    print(cumulative)


if __name__ == "__main__":
    solve()
