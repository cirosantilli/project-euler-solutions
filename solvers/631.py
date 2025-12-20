#!/usr/bin/env python3
"""
Project Euler 631: Constrained Permutations

We count permutations P of length <= n with:
  - no occurrence of the pattern 1243
  - at most m occurrences of the pattern 21 (i.e. inversions)

We need f(10^18, 40) mod 1_000_000_007.
"""

from collections import defaultdict

MOD = 1_000_000_007


def _exact_counts_up_to(limit: int, m: int, mod: int = MOD) -> list[int]:
    """
    Return g[0..limit], where g[i] is the number of valid permutations of length exactly i
    (valid = avoids 1243 and has <= m inversions), computed modulo mod.

    Technique:
      - Use the reverse-complement symmetry to count 2134-avoiding instead of 1243-avoiding.
      - Build permutations by choosing the *value of the last element* in standardized form.

    Encoding:
      Start with length 0. For i = 1..limit choose x in [1..i] meaning:
        take a permutation of {1..i-1}, increase all values >= x by 1, then append x.
      This is a bijection with permutations of size i.

    State:
      dp[(j, k, inv)] = number of prefixes of current length where
        j = minimum value that has appeared as the '2' in a 21 pattern (an inversion),
            or INF if no inversion seen yet.
        k = minimum value that has appeared as the '3' in a 213 pattern,
            or INF if no 213 seen yet.
        inv = inversion count so far (capped by m).
      Avoiding 2134 is enforced by never appending a value larger than the current k
      (after relabeling), because 2134 = 213 followed later by a larger element.
    """
    # INF must be larger than any label we will ever see (<= limit).
    INF = limit + 10

    # length 0: empty permutation
    dp: dict[tuple[int, int, int], int] = {(INF, INF, 0): 1}
    exact = [0] * (limit + 1)
    exact[0] = 1

    for i in range(1, limit + 1):
        ndp: defaultdict[tuple[int, int, int], int] = defaultdict(int)

        for (j, k, inv), cnt in dp.items():
            rem = m - inv
            # inversion add = i - x, so require i - x <= rem  =>  x >= i - rem
            x_min = 1 if rem >= i - 1 else i - rem

            # If we already have a 213 with minimum '3' = k, then x must not exceed it.
            # (A subtlety: existing labels may shift by +1 if they are >= x; the bound
            # below still remains valid and is a helpful pruning.)
            x_max = i if k == INF else min(i, k)

            if x_min > x_max:
                continue

            for x in range(x_min, x_max + 1):
                # Relabel existing tracked minima because inserting rank x pushes all
                # old values >= x up by 1.
                if j != INF and j >= x:
                    j_map = j + 1
                else:
                    j_map = j

                if k != INF and k >= x:
                    k_map = k + 1
                else:
                    k_map = k

                # Enforce "no 2134 created now": we can't append x if it's larger than
                # an already-existing minimum '3' of a 213 pattern (after relabeling).
                if k_map != INF and x > k_map:
                    continue

                new_inv = inv + (i - x)
                if new_inv > m:
                    continue

                # New inversions are formed with the appended x as the '1':
                # every value > x appears before it, and the smallest such '2' is x+1.
                if x < i:
                    if j_map == INF:
                        j_new = x + 1
                    else:
                        j_new = j_map if j_map < x + 1 else (x + 1)
                else:
                    j_new = j_map

                # New 213 patterns are formed if there is an inversion with '2' < x.
                # It's enough to compare with the minimum such '2' (j_map).
                k_new = k_map
                if j_map != INF and x > j_map:
                    if k_new == INF or x < k_new:
                        k_new = x

                ndp[(j_new, k_new, new_inv)] = (
                    ndp[(j_new, k_new, new_inv)] + cnt
                ) % mod

        dp = ndp
        exact[i] = sum(dp.values()) % mod

    return exact


def f(n: int, m: int, mod: int = MOD) -> int:
    """
    Compute f(n,m) = number of valid permutations of length at most n (mod mod).

    For large n, we use the fact that the number of valid permutations of *exact* length L
    stabilizes once L > 3m, so the prefix sum becomes linear after 3m.
    """
    if n < 0:
        return 0

    threshold = 3 * m  # beyond this, exact-length counts are constant
    limit = n if n <= threshold else (threshold + 1)

    exact = _exact_counts_up_to(limit, m, mod)

    if n <= threshold:
        return sum(exact[: n + 1]) % mod

    # n > threshold, need:
    # sum_{k=0..threshold} exact[k]  +  (n-threshold) * exact[threshold+1]
    base = sum(exact[: threshold + 1]) % mod
    c = exact[threshold + 1]
    return (base + ((n - threshold) % mod) * c) % mod


def solve() -> None:
    # Problem statement checks
    assert f(2, 0, MOD) == 3
    assert f(4, 5, MOD) == 32
    assert f(10, 25, MOD) == 294_400

    print(f(10**18, 40, MOD))


if __name__ == "__main__":
    solve()
