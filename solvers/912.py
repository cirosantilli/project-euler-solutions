#!/usr/bin/env python3
"""Project Euler 912: Where are the Odds?

Let s_n be the n-th positive integer whose binary representation does not
contain the substring '111'.

Define F(N) = sum(n^2 for 1 <= n <= N where s_n is odd).
Compute F(10^16) modulo 1_000_000_007.
"""

MOD = 1_000_000_007


def build_dp_until(total_needed: int):
    """Build DP tables until the cumulative number of valid values >= total_needed.

    State: (rem, cons)
      - rem  : how many bits remain to be chosen (suffix length)
      - cons : consecutive trailing ones in the already-fixed prefix (0,1,2)

    Each state enumerates valid suffixes in lexicographic order (0 before 1).
    Besides counts, we store for *odd* completions (ending in bit 1):
      - odd_cnt  : how many
      - odd_sum  : sum of 1-based ranks within this state's list
      - odd_sum2 : sum of squared ranks
    All sums are stored modulo MOD; counts are exact Python integers.
    """

    dp_count = [[1, 1, 1]]
    dp_odd_cnt = [[0, 1, 1]]
    dp_odd_sum = [[0, 1, 1]]
    dp_odd_sum2 = [[0, 1, 1]]

    # Length 1 corresponds to rem = 0 and MSB fixed to 1 (cons=1).
    cumulative = dp_count[0][1]

    while cumulative < total_needed:
        prev = len(dp_count) - 1
        c0 = dp_count[prev][0]
        c0m = c0 % MOD

        row_count = [0, 0, 0]
        row_oc = [0, 0, 0]
        row_os = [0, 0, 0]
        row_os2 = [0, 0, 0]

        for cons in range(3):
            # Next bit = 0: resets trailing ones.
            oc0 = dp_odd_cnt[prev][0]
            os0 = dp_odd_sum[prev][0]
            os20 = dp_odd_sum2[prev][0]

            # Next bit = 1: allowed only if cons < 2.
            if cons < 2:
                count1 = dp_count[prev][cons + 1]
                oc1 = dp_odd_cnt[prev][cons + 1]
                os1 = dp_odd_sum[prev][cons + 1]
                os21 = dp_odd_sum2[prev][cons + 1]
            else:
                count1 = 0
                oc1 = 0
                os1 = 0
                os21 = 0

            row_count[cons] = c0 + count1
            row_oc[cons] = (oc0 + oc1) % MOD

            # Branch 1 ranks are shifted by c0.
            row_os[cons] = (os0 + os1 + oc1 * c0m) % MOD
            row_os2[cons] = (
                os20 + os21 + (2 * c0m % MOD) * os1 + oc1 * (c0m * c0m % MOD)
            ) % MOD

        dp_count.append(row_count)
        dp_odd_cnt.append(row_oc)
        dp_odd_sum.append(row_os)
        dp_odd_sum2.append(row_os2)

        cumulative += dp_count[len(dp_count) - 1][1]

    return dp_count, dp_odd_cnt, dp_odd_sum, dp_odd_sum2


def prefix_odd_aggregates(
    rem: int, cons: int, t: int, dp_count, dp_odd_cnt, dp_odd_sum, dp_odd_sum2
):
    """Return (odd_cnt, odd_sum, odd_sum2) for the first t strings in state(rem, cons).

    Ranks are 1-based within the state's lexicographic list.
    Results are modulo MOD.
    """
    if t <= 0:
        return 0, 0, 0
    total = dp_count[rem][cons]
    if t >= total:
        return dp_odd_cnt[rem][cons], dp_odd_sum[rem][cons], dp_odd_sum2[rem][cons]
    if rem == 0:
        # Only one completion; it is odd iff the last chosen bit was 1 (cons > 0).
        if cons > 0:
            return 1, 1, 1
        return 0, 0, 0

    c0 = dp_count[rem - 1][0]
    if t <= c0:
        return prefix_odd_aggregates(
            rem - 1, 0, t, dp_count, dp_odd_cnt, dp_odd_sum, dp_odd_sum2
        )

    # Take all strings starting with 0, then a prefix of those starting with 1.
    oc0 = dp_odd_cnt[rem - 1][0]
    os0 = dp_odd_sum[rem - 1][0]
    os20 = dp_odd_sum2[rem - 1][0]

    if cons == 2:
        raise ValueError("Prefix length exceeds state size")

    oc1, os1, os21 = prefix_odd_aggregates(
        rem - 1, cons + 1, t - c0, dp_count, dp_odd_cnt, dp_odd_sum, dp_odd_sum2
    )

    c0m = c0 % MOD
    oc = (oc0 + oc1) % MOD
    os = (os0 + os1 + oc1 * c0m) % MOD
    os2 = (os20 + os21 + (2 * c0m % MOD) * os1 + oc1 * (c0m * c0m % MOD)) % MOD
    return oc, os, os2


def compute_F(N: int, dp_count, dp_odd_cnt, dp_odd_sum, dp_odd_sum2) -> int:
    """Compute F(N) modulo MOD using prebuilt DP tables."""
    ans = 0
    prev = 0
    remaining = N

    # Length L corresponds to rem = L-1. MSB is fixed to 1, so cons starts at 1.
    max_rem = len(dp_count) - 1
    for rem in range(max_rem + 1):
        if remaining == 0:
            break

        block_total = dp_count[rem][1]
        take = block_total if block_total <= remaining else remaining

        if take == block_total:
            oc = dp_odd_cnt[rem][1]
            os = dp_odd_sum[rem][1]
            os2 = dp_odd_sum2[rem][1]
        else:
            oc, os, os2 = prefix_odd_aggregates(
                rem, 1, take, dp_count, dp_odd_cnt, dp_odd_sum, dp_odd_sum2
            )

        prev_mod = prev % MOD
        ans = (
            ans + oc * (prev_mod * prev_mod % MOD) + (2 * prev_mod % MOD) * os + os2
        ) % MOD

        prev += take
        remaining -= take

    return ans


def _is_valid_no_111(x: int) -> bool:
    """Return True if x's binary representation has no substring '111'."""
    consec = 0
    while x:
        if x & 1:
            consec += 1
            if consec == 3:
                return False
        else:
            consec = 0
        x >>= 1
    return True


def _brute_s(n: int) -> int:
    """Brute-force s_n (only used for tiny n in asserts)."""
    found = 0
    x = 1
    while True:
        if _is_valid_no_111(x):
            found += 1
            if found == n:
                return x
        x += 1


def main() -> None:
    target_N = 10**16
    dp_count, dp_odd_cnt, dp_odd_sum, dp_odd_sum2 = build_dp_until(target_N)

    # Test values from the problem statement:
    assert _brute_s(1) == 1
    assert _brute_s(7) == 8
    assert compute_F(10, dp_count, dp_odd_cnt, dp_odd_sum, dp_odd_sum2) == 199

    print(compute_F(target_N, dp_count, dp_odd_cnt, dp_odd_sum, dp_odd_sum2))


if __name__ == "__main__":
    main()
