#!/usr/bin/env python3
"""Project Euler 939 - Partisan Nim

Compute E(N) mod 1234567891 for N=5000.
No external libraries are used.

The key reduction (from combinatorial game analysis) turns the win-condition into a
simple inequality involving partition statistics, allowing an O(N^2) DP solution.
"""

from array import array

MOD = 1234567891


def _build_partition_stats(N: int, mod: int):
    """Return (cnt0, cnt1, start) where:

    For each total n (0..N) and v (0..n), cntp[start[n] + v] is the number of
    partitions of n whose statistic v = n - (#parts) and whose parity of odd parts
    is p (0 or 1), taken modulo mod.

    start[n] is the offset into the flattened triangular arrays.

    DP over number of parts k:
        f[n,k,r] = f[n-1,k-1,r^1] + f[n-k,k,r^(k&1)]
    where r is parity of the number of odd parts.
    """

    # Triangular offsets: start[n] = n(n+1)/2
    start = [0] * (N + 1)
    for n in range(1, N + 1):
        start[n] = start[n - 1] + n

    total_entries = (N + 1) * (N + 2) // 2
    cnt0 = array("I", [0]) * total_entries
    cnt1 = array("I", [0]) * total_entries

    # Empty partition of 0
    cnt0[0] = 1

    prev0 = array("I", [0]) * (N + 1)
    prev1 = array("I", [0]) * (N + 1)
    prev0[0] = 1  # f[0,0,0] = 1

    for k in range(1, N + 1):
        cur0 = array("I", [0]) * (N + 1)
        cur1 = array("I", [0]) * (N + 1)

        if (k & 1) == 0:
            # parity unchanged in the (n-k, k) branch
            for n in range(k, N + 1):
                a0 = prev1[n - 1]
                a1 = prev0[n - 1]
                b0 = cur0[n - k]
                b1 = cur1[n - k]

                s0 = a0 + b0
                if s0 >= mod:
                    s0 -= mod
                s1 = a1 + b1
                if s1 >= mod:
                    s1 -= mod

                cur0[n] = s0
                cur1[n] = s1

                v = n - k
                idx = start[n] + v

                t = cnt0[idx] + s0
                if t >= mod:
                    t -= mod
                cnt0[idx] = t

                t = cnt1[idx] + s1
                if t >= mod:
                    t -= mod
                cnt1[idx] = t
        else:
            # parity flips in the (n-k, k) branch
            for n in range(k, N + 1):
                a0 = prev1[n - 1]
                a1 = prev0[n - 1]
                b0 = cur1[n - k]
                b1 = cur0[n - k]

                s0 = a0 + b0
                if s0 >= mod:
                    s0 -= mod
                s1 = a1 + b1
                if s1 >= mod:
                    s1 -= mod

                cur0[n] = s0
                cur1[n] = s1

                v = n - k
                idx = start[n] + v

                t = cnt0[idx] + s0
                if t >= mod:
                    t -= mod
                cnt0[idx] = t

                t = cnt1[idx] + s1
                if t >= mod:
                    t -= mod
                cnt1[idx] = t

        prev0, prev1 = cur0, cur1

    return cnt0, cnt1, start


def compute_E(N: int, mod: int = MOD) -> int:
    """Compute E(N) modulo mod."""

    cnt0, cnt1, start = _build_partition_stats(N, mod)

    # cum0[v] / cum1[v] = sum_{b <= m} cntp[b][v]  (mod mod)
    cum0 = array("I", [0]) * (N + 1)
    cum1 = array("I", [0]) * (N + 1)

    # prefix sums over v for current cum arrays (only first (a+1) used each step)
    pref0 = [0] * (N + 1)
    pref1 = [0] * (N + 1)

    ans = 0

    # We iterate m = max stones on B side. For each m we include all b<=m,
    # and pair with a = N - m.
    for m in range(0, N + 1):
        sb = start[m]

        # Add distributions for total b = m into cumulative B-side counts.
        for v in range(0, m + 1):
            idx = sb + v

            x = cnt0[idx]
            if x:
                s = cum0[v] + x
                if s >= mod:
                    s -= mod
                cum0[v] = s

            x = cnt1[idx]
            if x:
                s = cum1[v] + x
                if s >= mod:
                    s -= mod
                cum1[v] = s

        a = N - m
        sa = start[a]

        # Build prefix sums of B distributions only up to v=a, since thresholds
        # vA-1 and vA-2 never exceed a-1.
        run = 0
        for v in range(0, a + 1):
            run += cum0[v]
            if run >= mod:
                run -= mod
            pref0[v] = run

        run = 0
        for v in range(0, a + 1):
            run += cum1[v]
            if run >= mod:
                run -= mod
            pref1[v] = run

        # Count A partitions of total a against all B partitions of total <= m.
        # Win condition for A reduces to:
        #   Let M = vA - vB and S = parityOddA XOR parityOddB.
        #   If S=0 require M>=1 (vB <= vA-1)
        #   If S=1 require M>=2 (vB <= vA-2)
        for vA in range(0, a + 1):
            idxA = sa + vA
            a0 = cnt0[idxA]
            a1 = cnt1[idxA]

            x_same = vA - 1
            x_diff = vA - 2

            if a0:
                s = 0
                if x_same >= 0:
                    s += pref0[x_same]
                if x_diff >= 0:
                    s += pref1[x_diff]
                if s >= mod:
                    s -= mod
                ans = (ans + a0 * s) % mod

            if a1:
                t = 0
                if x_same >= 0:
                    t += pref1[x_same]
                if x_diff >= 0:
                    t += pref0[x_diff]
                if t >= mod:
                    t -= mod
                ans = (ans + a1 * t) % mod

    return ans


def main() -> None:
    # Test value from the problem statement:
    assert compute_E(4, MOD) == 9

    # Required output:
    print(compute_E(5000, MOD))


if __name__ == "__main__":
    main()
