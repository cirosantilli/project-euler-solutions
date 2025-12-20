#!/usr/bin/env python3
"""Project Euler 951: A Game of Chance

We count how many starting configurations are *fair* (first player wins with
probability 1/2).

Key observation (explained in README): the game outcome depends only on the
parity of the number of turns. Turns across maximal same-colour runs are
independent, and only runs of length 2 make the overall parity unbiased.
So we count balanced R/B strings of length 2n that contain at least one
maximal run of length 2.

This file prints F(26) by default.
"""

from __future__ import annotations

import sys


def comb(n: int, k: int) -> int:
    """Compute n choose k (integer)."""
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    num = 1
    den = 1
    for i in range(1, k + 1):
        num *= n - (k - i)
        den *= i
    return num // den


def count_balanced_no_run2(n: int) -> int:
    """Count length 2n strings with n R, n B and *no maximal run of length 2*.

    We build the string left-to-right, tracking:
      - r: number of R used so far (0..n), B count is implied by position
      - last: last colour (0=R, 1=B)
      - rs: current run length category: 1, 2, or 3 (meaning >=3)

    Rule "no run of length 2" becomes: you may NOT switch colours when rs==2.
    Final state also must avoid rs==2.
    """

    if n <= 0:
        return 1 if n == 0 else 0

    L = 2 * n

    # dp[r][last][rs]
    dp = [[[0, 0, 0, 0] for _ in range(2)] for _ in range(n + 1)]
    # first card
    dp[1][0][1] = 1  # start with R
    dp[0][1][1] = 1  # start with B

    for pos in range(1, L):
        new = [[[0, 0, 0, 0] for _ in range(2)] for _ in range(n + 1)]
        for r in range(0, n + 1):
            b = pos - r
            if b < 0 or b > n:
                continue
            for last in (0, 1):
                for rs in (1, 2, 3):
                    val = dp[r][last][rs]
                    if not val:
                        continue

                    # Add an R (0)
                    if r + 1 <= n:
                        if last == 0:
                            nrs = rs + 1 if rs < 3 else 3
                            new[r + 1][0][nrs] += val
                        else:
                            # switching B -> R forbidden if ending a run of length 2
                            if rs != 2:
                                new[r + 1][0][1] += val

                    # Add a B (1)
                    if b + 1 <= n:
                        if last == 1:
                            nrs = rs + 1 if rs < 3 else 3
                            new[r][1][nrs] += val
                        else:
                            if rs != 2:
                                new[r][1][1] += val
        dp = new

    # Accept balanced states (r=n, b=n) where the final run is not length 2.
    total = 0
    r = n
    for last in (0, 1):
        total += dp[r][last][1]
        total += dp[r][last][3]
    return total


def F(n: int) -> int:
    """Number of fair starting configurations for the given n."""
    total = comb(2 * n, n)
    unfair = count_balanced_no_run2(n)
    return total - unfair


def _self_test() -> None:
    # Test values given in the problem statement
    assert F(2) == 4
    assert F(8) == 11892


def main(argv: list[str]) -> None:
    _self_test()

    n = 26
    if len(argv) >= 2:
        n = int(argv[1])

    print(F(n))


if __name__ == "__main__":
    main(sys.argv)
