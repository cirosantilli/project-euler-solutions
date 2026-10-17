#!/usr/bin/env python

from itertools import combinations
from math import comb


def intersection_sizes(v: int, k: int, t: int) -> set[int]:
    """Possible intersections with a block of a Steiner system S(t, k, v)."""
    counts = [0] * t
    for j in range(t - 1, -1, -1):
        numerator = comb(v - j, t - j)
        denominator = comb(k - j, t - j)
        if numerator % denominator:
            raise ValueError("Nonintegral Steiner system parameters")
        counts[j] = comb(k, j) * (numerator // denominator - 1) - sum(
            comb(i, j) * counts[i] for i in range(j + 1, t)
        )
    if min(counts) < 0:
        raise ValueError("Impossible Steiner system parameters")
    return {i for i, count in enumerate(counts) if count} | {k}


def solve(
    v: int, k: int, t: int, nights: tuple[str, ...], target: str
) -> str:
    """Find the unique target block forced by the known block intersections."""
    allowed = intersection_sizes(v, k, t)
    wanted = sum(1 << (ord(member) - ord("A")) for member in target)
    known = [sum(1 << (ord(c) - ord("A")) for c in night) for night in nights]
    remaining = [i for i in range(v) if not wanted >> i & 1]
    answer = None
    for extra in combinations(remaining, k - len(target)):
        candidate = wanted | sum(1 << i for i in extra)
        if all((candidate & block).bit_count() in allowed for block in known):
            if answer is not None:
                raise ValueError("The intersection constraints do not force one block")
            answer = candidate
    if answer is None:
        raise ValueError("No compatible block")
    return "".join(chr(ord("A") + i) for i in range(v) if answer >> i & 1)


if __name__ == "__main__":
    assert solve(7, 3, 2, ("ABC", "BDF"), "EG") == "BEG"
    nights = (
        "ABCDKMPR",
        "ABJKLMNW",
        "ACIOQTUW",
        "ADFMPQTU",
        "AEFNPTVX",
        "AEFQRSTW",
        "AEGIMQSU",
        "BDEKORUX",
        "BEFGHIKU",
        "DEFHKOSW",
        "FHIJKRST",
    )
    print(solve(24, 8, 5, nights, "CGJLV"))
