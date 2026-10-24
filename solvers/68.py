#!/usr/bin/env python
from itertools import combinations, permutations


def ring_string(outer: list[int], inner: tuple[int, ...], start: int) -> str:
    n = len(inner)
    parts: list[str] = []
    for offset in range(n):
        index = (start + offset) % n
        parts.append(str(outer[index]) + str(inner[index]) + str(inner[(index + 1) % n]))
    return "".join(parts)


def solve(n: int = 5, digits: int = 16) -> str:
    """
    Maximize a canonical magic n-gon string using labels 1 through 2*n.
    Return an empty string if no ring has the requested decimal length.

    @formalsw: Py from project_euler_statements.p68 import naive
    @formalsw: Returns
    @formalsw: Equiv(naive)
    """
    if n < 3 or digits < 3 * n:
        return ""
    labels = list(range(1, 2 * n + 1))
    nums = set(labels)
    total_digits = sum(len(str(value)) for value in range(1, 2 * n + 1))
    total = n * (2 * n + 1)
    best = ""

    for inner_set in combinations(labels, n):
        # Inner labels occur twice; outer labels occur once in the text.
        if total_digits + sum(len(str(value)) for value in inner_set) != digits:
            continue
        inner_sum = sum(inner_set)
        if (total + inner_sum) % n != 0:
            continue
        target = (total + inner_sum) // n
        remaining_outer = nums - set(inner_set)

        for inner in permutations(inner_set):
            outer: list[int] = []
            for index in range(n):
                value = target - inner[index] - inner[(index + 1) % n]
                if value < 1 or value > 2 * n:
                    break
                outer.append(value)
            if len(outer) != n:
                continue
            if set(outer) != remaining_outer or len(set(outer)) != n:
                continue
            start = min(list(range(n)), key=lambda index: outer[index])
            text = ring_string(outer, inner, start)
            if len(text) == digits and text > best:
                best = text
    return best


if __name__ == "__main__":
    assert solve(3, 9) == "432621513"
    print(solve())
