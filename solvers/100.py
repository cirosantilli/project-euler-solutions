#!/usr/bin/env python


def next_solution(u: int, v: int) -> tuple[int, int]:
    """
    Given (u, v) solving u^2 - 2*v^2 = -1, generate the next larger solution.
    Uses multiplication by (3 + 2*sqrt(2)).
    """
    return 3 * u + 4 * v, 2 * u + 3 * v


def blue_red_for_total_limit(limit: int) -> tuple[int, int, int]:
    """
    Find the first arrangement with at least two discs and total n > limit such that:
        P(two blue) = b/n * (b-1)/(n-1) = 1/2
    Returns (b, r, n).
    """
    # Use u = 2n - 1, v = 2b - 1, then u^2 - 2*v^2 = -1.
    u, v = 7, 5  # The first valid arrangement has three blue discs out of four.

    while True:
        n = (u + 1) // 2
        b = (v + 1) // 2
        if n > limit:
            r = n - b
            return b, r, n
        u, v = next_solution(u, v)


def solve(limit: int) -> int:
    return blue_red_for_total_limit(limit)[0]


if __name__ == "__main__":
    # Generate early known solutions:
    u, v = 1, 1
    sols = []
    for _ in range(4):
        n = (u + 1) // 2
        b = (v + 1) // 2
        sols.append((b, n))
        u, v = next_solution(u, v)

    # (b, n) sequence starts: (1,1), (3,4), (15,21), (85,120), ...
    assert sols[2] == (15, 21)
    assert sols[3] == (85, 120)
    assert solve(3) == 3
    assert solve(4) == 15

    print(solve(10**12))
