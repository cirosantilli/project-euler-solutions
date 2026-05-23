#!/usr/bin/env python

from collections import defaultdict
from functools import lru_cache


@lru_cache(maxsize=None)
def transitions(state):
    out = defaultdict(int)
    length = len(state)
    i = 0

    while i < length:
        seg_len = state[i]
        j = i + 1
        while j < length and state[j] == seg_len:
            j += 1
        count = j - i
        base = state[:i] + state[i + 1 :]

        if seg_len == 1:
            out[base] += count
        else:
            out[tuple(sorted(base + (seg_len - 1,)))] += 2 * count

            rest = seg_len - 1
            for left in range(1, (rest // 2) + 1):
                right = rest - left
                mult = 1 if left == right else 2
                out[tuple(sorted(base + (left, right)))] += mult * count

        i = j

    return tuple(out.items())


@lru_cache(maxsize=None)
def expected_max(state, best):
    if not state:
        return float(best)

    remaining = sum(state)
    total = 0.0
    for next_state, weight in transitions(state):
        next_best = best if best >= len(next_state) else len(next_state)
        total += weight * expected_max(next_state, next_best)
    return total / remaining


def solve(n=40):
    return f"{expected_max((n,), 1):.6f}"


if __name__ == "__main__":
    assert f"{expected_max((10,), 1):.12f}" == "3.400731922399"
    print(solve())
