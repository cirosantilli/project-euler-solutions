#!/usr/bin/env python3
"""
Project Euler 823: Factor Shuffle

No external dependencies. Single-threaded.

We represent each number as its sorted list of prime factors (with multiplicity).
A "round" removes the smallest prime factor from each list, concatenates those removed
primes into a new list (sorted), appends that list, and drops any emptied lists.

For large m, the process reaches a regime where the k-th prime factor of the newly
created number becomes periodic with period k. After detecting that regime, we can
jump to extremely large m by indexing into these short cycles.
"""

from __future__ import annotations

from collections import deque
import math


MOD = 1234567891


def sieve_spf(n: int) -> list[int]:
    """Smallest prime factor sieve for 0..n."""
    spf = list(range(n + 1))
    limit = int(n**0.5)
    for i in range(2, limit + 1):
        if spf[i] == i:  # prime
            step = i
            start = i * i
            for j in range(start, n + 1, step):
                if spf[j] == j:
                    spf[j] = i
    return spf


def factor_list(x: int, spf: list[int]) -> list[int]:
    """Prime factors of x in nondecreasing order."""
    out: list[int] = []
    while x > 1:
        p = spf[x]
        out.append(p)
        x //= p
    return out


def direct_sum(n: int, m: int) -> int:
    """
    Exact simulation (integer arithmetic), intended for small n,m examples.
    Returns S(n,m) as an integer (not reduced mod).
    """
    spf = sieve_spf(n)
    piles = [factor_list(i, spf) for i in range(2, n + 1)]
    pos = [0] * len(piles)

    for _ in range(m):
        k = len(piles)
        extracted = [0] * k

        new_piles: list[list[int]] = []
        new_pos: list[int] = []
        ap = new_piles.append
        bp = new_pos.append

        for idx in range(k):
            f = piles[idx]
            p = pos[idx]
            extracted[idx] = f[p]
            p += 1
            if p < len(f):
                ap(f)
                bp(p)

        extracted.sort()
        ap(extracted)
        bp(0)
        piles, pos = new_piles, new_pos

    total = 0
    for f, p in zip(piles, pos):
        prod = 1
        for v in f[p:]:
            prod *= v
        total += prod
    return total


def simulate_until_periodic(
    n: int,
    *,
    k_extra: int = 10,
    streak_needed: int = 2000,
    max_rounds: int = 200000,
) -> tuple[int, list[list[int]], int]:
    """
    Simulate the factor-shuffle process until the "column periodicity" condition holds
    for many consecutive rounds.

    Returns:
      end_t: round index when the periodicity was accepted
      patterns: patterns[k] is the cycle (list length k) for the k-th prime factor
                of the added number, aligned so that for any u > end_t:
                x(u,k) = patterns[k][(u - end_t - 1) % k]
                (x(u,k) is the k-th smallest prime factor of the number added at round u,
                 or 1 if that number has fewer than k prime factors)
      kmax: largest k such that patterns[k] contains a value > 1
    """
    spf = sieve_spf(n)

    piles: list[list[int]] = []
    pos: list[int] = []
    total_factors = 0
    for i in range(2, n + 1):
        f = factor_list(i, spf)
        piles.append(f)
        pos.append(0)
        total_factors += len(f)

    # A safe upper bound on the typical number of prime factors in the "new" number.
    k_lim = int(math.isqrt(2 * total_factors)) + k_extra

    bufs: list[deque[int] | None] = [None] * (k_lim + 1)
    for k in range(1, k_lim + 1):
        bufs[k] = deque(maxlen=k)

    stable_streak = 0
    t = 0

    while t < max_rounds:
        t += 1
        k = len(piles)

        extracted = [0] * k
        new_piles: list[list[int]] = []
        new_pos: list[int] = []
        ap = new_piles.append
        bp = new_pos.append

        for idx in range(k):
            f = piles[idx]
            p = pos[idx]
            extracted[idx] = f[p]
            p += 1
            if p < len(f):
                ap(f)
                bp(p)

        extracted.sort()
        ap(extracted)
        bp(0)
        piles, pos = new_piles, new_pos

        # Update per-column buffers and check whether each column is now rotating perfectly.
        # For column k, "rotating perfectly" this round means the newly appended value equals
        # the value from k rounds ago (the leftmost element in the length-k buffer).
        if t <= k_lim:
            # Not all buffers are full yet: just fill them.
            mm = min(len(extracted), k_lim)
            for kk in range(1, mm + 1):
                bufs[kk].append(extracted[kk - 1])  # type: ignore[union-attr]
            for kk in range(mm + 1, k_lim + 1):
                bufs[kk].append(1)  # type: ignore[union-attr]
            stable_streak = 0
            continue

        all_ok = True
        mm = min(len(extracted), k_lim)

        # Columns that exist in this round (kk <= len(extracted))
        for kk in range(1, mm + 1):
            v = extracted[kk - 1]
            b = bufs[kk]  # type: ignore[assignment]
            if b[0] != v:
                all_ok = False
            b.append(v)

        # Columns beyond the new number's length: treat as 1
        for kk in range(mm + 1, k_lim + 1):
            b = bufs[kk]  # type: ignore[assignment]
            if b[0] != 1:
                all_ok = False
            b.append(1)

        if all_ok:
            stable_streak += 1
            if stable_streak >= streak_needed:
                patterns: list[list[int]] = [[] for _ in range(k_lim + 1)]
                kmax = 0
                for kk in range(1, k_lim + 1):
                    pat = list(bufs[kk])  # type: ignore[arg-type]
                    patterns[kk] = pat
                    if kmax < kk and any(x != 1 for x in pat):
                        kmax = kk
                return t, patterns, kmax
        else:
            stable_streak = 0

    raise RuntimeError("Periodicity was not detected within max_rounds.")


def sum_at_round_mod(n: int, m: int, mod: int) -> int:
    """Compute S(n,m) modulo mod."""
    # For very small m, a direct simulation is simplest.
    if n <= 50 and m <= 5000:
        return direct_sum(n, m) % mod

    end_t, patterns, kmax = simulate_until_periodic(n)

    # If m is before the detected periodic regime, fall back to a direct mod simulation.
    if m <= end_t:
        # Mod-only simulation (still factor-based, but avoids huge integers).
        spf = sieve_spf(n)
        piles = [factor_list(i, spf) for i in range(2, n + 1)]
        pos = [0] * len(piles)
        for _ in range(m):
            k = len(piles)
            extracted = [0] * k
            new_piles: list[list[int]] = []
            new_pos: list[int] = []
            ap = new_piles.append
            bp = new_pos.append
            for idx in range(k):
                f = piles[idx]
                p = pos[idx]
                extracted[idx] = f[p]
                p += 1
                if p < len(f):
                    ap(f)
                    bp(p)
            extracted.sort()
            ap(extracted)
            bp(0)
            piles, pos = new_piles, new_pos

        total = 0
        for f, p in zip(piles, pos):
            prod = 1
            for v in f[p:]:
                prod = (prod * v) % mod
            total = (total + prod) % mod
        return total

    # In the periodic regime, the k-th prime factor (with 1 padding) follows:
    # x(u,k) = patterns[k][(u - end_t - 1) % k]
    #
    # After m rounds, every surviving number comes from a recently added number:
    # the one added at round (m - d) has been divided d times, leaving the suffix
    # product of its prime factors from index (d+1) onward.
    r0 = m - end_t - 1
    total = 0

    # Age d ranges 0..kmax-1; if the (d+1)-th factor is 1, that number has already vanished.
    for d in range(kmax):
        r = r0 - d
        if patterns[d + 1][r % (d + 1)] == 1:
            continue

        prod = 1
        # Multiply remaining factors (ignoring 1s).
        for k in range(kmax, d, -1):
            v = patterns[k][r % k]
            if v != 1:
                prod = (prod * v) % mod
        total = (total + prod) % mod

    return total


def main() -> None:
    # Examples from the problem statement:
    assert direct_sum(5, 3) == 21
    assert direct_sum(10, 100) == 257

    n = 10_000
    m = 10**16
    print(sum_at_round_mod(n, m, MOD))


if __name__ == "__main__":
    main()
