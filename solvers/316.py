#!/usr/bin/env python

POW10 = [10**i for i in range(20)]


def prefix_function(word: str) -> list[int]:
    pi = [0] * len(word)
    matched = 0

    for i in range(1, len(word)):
        while matched and word[i] != word[matched]:
            matched = pi[matched - 1]
        if word[i] == word[matched]:
            matched += 1
        pi[i] = matched

    return pi


def g(n: int) -> int:
    word = str(n)
    pi = prefix_function(word)

    expected_end = 0
    length = len(word)
    border = length
    while border:
        expected_end += POW10[border]
        border = pi[border - 1]

    return expected_end - length + 1


def compute(limit: int, divider: int) -> int:
    return sum(g(divider // n) for n in range(2, limit + 1))


if __name__ == "__main__":
    assert g(535) == 1008
    assert compute(999, 10**6) == 27280188
    print(compute(999999, 10**16))
