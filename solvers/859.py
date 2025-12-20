#!/usr/bin/env python3
"""Project Euler 859: Cookie Game

Odd (Left) and Even (Right) play on piles of cookies.
A move chooses a single pile and replaces it by two equal smaller piles
(after eating 1 cookie for Odd, or 2 cookies for Even).

We count C(N): the number of initial (unordered) pile partitions of N
for which Even has a winning strategy (Odd starts).

No external libraries are used.
"""

from __future__ import annotations


def pile_value_table(n_max: int) -> list[int]:
    """Compute the combinatorial game value g(n) for a single pile of size n.

    Interpret Odd as Left and Even as Right in normal-play combinatorial game theory.

    For a pile of size n:
      - if n is odd, Left has the only move to two piles of size (n-1)/2
      - if n is even, Right has the only move to two piles of size (n-2)/2

    Each pile is a *number* (cold game). Moreover, these numbers are integers.

    Let g(n) be the value (advantage for Odd/Left). Then:
      g(0) = 0
      if n = 2m+1 (odd): option is 2*g(m)
        - if 2*g(m) < 0, the simplest number > option is 0
        - else the simplest number > option is 2*g(m) + 1
      if n = 2m (even): option is 2*g(m-1)
        - if 2*g(m-1) > 0, the simplest number < option is 0
        - else the simplest number < option is 2*g(m-1) - 1

    The resulting g(n) is an integer for all n.
    """

    g = [0] * (n_max + 1)
    g[0] = 0

    for n in range(1, n_max + 1):
        if n & 1:
            # n = 2m+1
            m = n >> 1
            option = 2 * g[m]
            g[n] = 0 if option < 0 else option + 1
        else:
            # n = 2m
            m = n >> 1
            option = 2 * g[m - 1]
            g[n] = 0 if option > 0 else option - 1

    return g


def count_even_wins(n: int) -> int:
    """Return C(n): number of unordered partitions of n where Even wins.

    For any position with piles p1, p2, ... the total game value is
    g(p1)+g(p2)+... .

    Odd (Left) wins iff total value > 0.
    Even (Right) wins iff total value <= 0 (including 0 where the second
    player wins).

    Therefore, we count partitions of n whose summed pile values are <= 0.

    We compute a 2D partition DP:
      dp[cookies][value] = number of partitions summing to `cookies`
                           with total value `value`.

    Value bounds:
      - each pile contributes at least -size/2 (for even sizes)
      - each pile contributes at most +size
      => total value is in [ -n//2 , n ].
    """

    g = pile_value_table(n)

    v_min = -(n // 2)
    v_max = n
    width = v_max - v_min + 1
    offset = -v_min  # value 0 is at index offset

    dp = [[0] * width for _ in range(n + 1)]
    dp[0][offset] = 1

    # Standard "coin change" DP to count unordered partitions:
    # iterate pile sizes increasing so each multiset is counted once.
    for size in range(1, n + 1):
        dv = g[size]

        if dv >= 0:
            # dp[c][v] += dp[c-size][v-dv]
            for cookies in range(size, n + 1):
                src = dp[cookies - size]
                dst = dp[cookies]
                for vi in range(dv, width):
                    dst[vi] += src[vi - dv]
        else:
            # dv < 0: dp[c][v] += dp[c-size][v - dv] = dp[c-size][v + (-dv)]
            shift = -dv
            for cookies in range(size, n + 1):
                src = dp[cookies - size]
                dst = dp[cookies]
                # Need vi + shift < width
                limit = width - shift
                for vi in range(0, limit):
                    dst[vi] += src[vi + shift]

    # Sum counts with total value <= 0.
    return sum(dp[n][: offset + 1])


def main() -> None:
    # Test values from the problem statement.
    assert count_even_wins(5) == 2
    assert count_even_wins(16) == 64

    print(count_even_wins(300))


if __name__ == "__main__":
    main()
