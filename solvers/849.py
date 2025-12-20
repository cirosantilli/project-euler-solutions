#!/usr/bin/env python3
"""Project Euler 849 - The Tournament

We count distinct final point-multisets (teams indistinguishable) for a double
round-robin with scoring 2/1/0.

The implementation follows a local-constraint reformulation and a dynamic
programming count modulo 1_000_000_007.
"""

import sys

MOD = 1_000_000_007


def count_outcomes(n: int, mod: int = MOD) -> int:
    """Return F(n) modulo mod."""
    if n <= 0:
        return 0
    if n == 1:
        return 1

    # From the Landau-type characterization for 4-point tournaments, F(n)
    # equals the number of nonnegative integer sequences p_0..p_n with
    # p_0 = p_n = 0 and 2 p_i - (p_{i-1} + p_{i+1}) <= 4 for i=1..n-1.
    #
    # Let d_i = p_i - p_{i-1}. Then the inequality becomes:
    #   d_{i+1} >= d_i - 4.
    # We DP over (height p_i, last difference d_i).

    vmax = 2 * (n - 1)  # tight bound for |d_i| in any valid sequence
    off = vmax
    V = 2 * vmax + 1  # number of possible d values

    # Tight height bound at each position i: p_i <= 2*i*(n-i)
    maxh = [0] * (n + 1)
    for i in range(n + 1):
        maxh[i] = 2 * i * (n - i)

    # dp[h][vi] where vi encodes d = vi-off, after i steps.
    # Start at i=1: p_1 = d_1 (since p_0=0), and d_1 >= 0.
    dp = [[0] * V for _ in range(maxh[1] + 1)]
    for d in range(0, vmax + 1):
        dp[d][d + off] = 1

    for step in range(1, n):
        mh = maxh[step]
        mh_next = maxh[step + 1]
        nxt = [[0] * V for _ in range(mh_next + 1)]

        # For fixed height h and a velocity-index vi representing threshold t,
        # prefix sums over d<=t are accumulated in cum. That cum equals the
        # number of ways to choose next d' = t-4 (because d' needs all previous
        # d <= d'+4). This lets us update in O(V) per height, not O(V^2).
        for h in range(mh + 1):
            row = dp[h]
            cum = 0

            # Updating is valid only when the next height h' is in [0, mh_next].
            # For vi>=4, we add to next state (h' = h + (vi-off-4), d' = vi-off-4).
            base = h - off - 4
            vi_start = -base
            if vi_start < 4:
                vi_start = 4
            if vi_start > V:
                vi_start = V
            vi_end = mh_next - base
            if vi_end > V - 1:
                vi_end = V - 1

            # cum for vi < vi_start
            for vi in range(vi_start):
                cum += row[vi]
                if cum >= mod:
                    cum -= mod

            # cum + updates for vi in [vi_start, vi_end]
            if vi_start <= vi_end:
                for vi in range(vi_start, vi_end + 1):
                    cum += row[vi]
                    if cum >= mod:
                        cum -= mod
                    h2 = base + vi  # in range by construction
                    idx = vi - 4
                    val = nxt[h2][idx] + cum
                    if val >= mod:
                        val -= mod
                    nxt[h2][idx] = val

                # finish cum for vi > vi_end
                for vi in range(vi_end + 1, V):
                    cum += row[vi]
                    if cum >= mod:
                        cum -= mod
            else:
                # no update region
                for vi in range(vi_start, V):
                    cum += row[vi]
                    if cum >= mod:
                        cum -= mod

            total = cum  # sum over all d

            # For the largest 4 velocities, the threshold d'+4 saturates at vmax,
            # so they all receive the full sum 'total'.
            if total:
                base_h = h - off
                for vi in range(V - 4, V):
                    h2 = base_h + vi
                    if 0 <= h2 <= mh_next:
                        val = nxt[h2][vi] + total
                        if val >= mod:
                            val -= mod
                        nxt[h2][vi] = val

        dp = nxt

    # At step n, height must be 0; sum over all last differences.
    return sum(dp[0]) % mod


def main() -> None:
    # Test values from the problem statement.
    assert count_outcomes(2) == 3
    assert count_outcomes(7) == 32923

    print(count_outcomes(100))


if __name__ == "__main__":
    main()
