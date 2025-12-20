#!/usr/bin/env python3
"""
Project Euler 923: Young's Game B

We count S(m, w): the number of ordered sequences of m staircases (a,b,k) with
a,b,k >= 1 and a+b+k <= w for which Right (the horizontal mover, who moves first)
wins under optimal play.

No external libraries are used (only the Python standard library).
"""

from __future__ import annotations

from collections import defaultdict


MOD = 1_000_000_007


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def reduced_hook(a: int, b: int, k: int) -> tuple[int, int]:
    """
    Return (M, N) for the reduced Young diagram of an (a,b,k)-staircase,
    where the reduced diagram is always a hook of shape (M, 1^(N-1)).

    The (a,b,k)-staircase has k blocks; each block contributes 'a' rows, and
    the row lengths are:
        (k*b) repeated a times,
        ((k-1)*b) repeated a times,
        ...
        (b) repeated a times.

    The reduction repeatedly removes the first row and first column until the
    Durfee size becomes 1. For a partition λ, the remaining hook is determined
    by the Durfee size d and the d-th row/column:
        M = λ_d - d + 1
        N = (#rows with λ_i >= d) - d + 1
    """
    # Compute Durfee size d = max r with λ_r >= r, exploiting the block structure.
    d = 0
    for j in range(k):  # block index
        row_len = (k - j) * b
        start = j * a + 1
        if row_len < start:
            continue
        end = (j + 1) * a
        cand = end if end < row_len else row_len
        if cand > d:
            d = cand
    # d >= 1 always for valid staircases
    block_of_row_d = (d - 1) // a
    lambda_d = (k - block_of_row_d) * b
    M = lambda_d - d + 1

    # Column height at column d: number of rows with length >= d
    need_blocks = ceil_div(d, b)  # smallest t with t*b >= d
    last_block = k - need_blocks  # max j such that (k-j)*b >= d
    col_height = (last_block + 1) * a
    N = col_height - d + 1

    # Sanity: reduced hook must be non-empty.
    if M <= 0 or N <= 0:
        raise ValueError(
            f"Invalid reduced hook for (a,b,k)=({a},{b},{k}): (M,N)=({M},{N}), d={d}"
        )
    return M, N


def classify_staircase(a: int, b: int, k: int) -> tuple[str, int | tuple[int, int]]:
    """
    Classify a staircase into either:
      - ("int", v) where v is an integer game value, or
      - ("hot", (t, R)) where the game is a (possibly zero-temperature) hot game
        with options {L | R}, L >= R integers, and:
            t = L - R   (temperature * 2 in some conventions)
            R = right option (chosen when Down plays on this component)
        Left option is L = R + t.

    For a reduced hook (M, 1^(N-1)):
      - N == 1: value is M-1
      - M == 1: value is -(N-1)
      - otherwise: value is {M-2 | -(N-2)}
    """
    M, N = reduced_hook(a, b, k)
    if N == 1:  # single row
        return "int", M - 1
    if M == 1:  # single column
        return "int", -(N - 1)

    L = M - 2
    R = -(N - 2)
    t = L - R  # = M + N - 4
    return "hot", (t, R)


def counts_for_w(w: int) -> tuple[dict[int, int], dict[tuple[int, int], int]]:
    """
    Enumerate all (a,b,k)-staircases with a+b+k <= w and count how many map to each:
      - integer value v  -> ints[v]
      - hot type (t, R)  -> hots[(t, R)]
    """
    ints: dict[int, int] = defaultdict(int)
    hots: dict[tuple[int, int], int] = defaultdict(int)

    for a in range(1, w - 1):
        for b in range(1, w - a):
            max_k = w - a - b
            if max_k < 1:
                continue
            for k in range(1, max_k + 1):
                typ, val = classify_staircase(a, b, k)
                if typ == "int":
                    ints[int(val)] += 1
                else:
                    hots[val] += 1
    return ints, hots


def solve(m: int, w: int) -> int:
    """
    Compute S(m, w) modulo MOD.
    """
    # factorials for converting multiset counts -> ordered sequence counts
    fact = [1] * (m + 1)
    for i in range(1, m + 1):
        fact[i] = fact[i - 1] * i % MOD
    invfact = [1] * (m + 1)
    invfact[m] = pow(fact[m], MOD - 2, MOD)
    for i in range(m, 0, -1):
        invfact[i - 1] = invfact[i] * i % MOD

    ints, hots = counts_for_w(w)

    # DP over "hot" components in descending temperature.
    # State: used_count, parity, sum_value -> coefficient in EGF form (product c^k / k!)
    # parity = 0 means Right to move when the next hot component is played,
    # parity = 1 means Down to move.
    dp_hot = [[defaultdict(int) for _ in range(2)] for _ in range(m + 1)]
    dp_hot[0][0][0] = 1

    hot_types = sorted(
        ((t, R, c) for (t, R), c in hots.items()),
        key=lambda x: (-x[0], x[1]),
    )

    for t, R, c in hot_types:
        # poly[k] = c^k / k!
        poly = [0] * (m + 1)
        poly[0] = 1
        p = 1
        for k in range(1, m + 1):
            p = p * c % MOD
            poly[k] = p * invfact[k] % MOD

        new = [[defaultdict(int) for _ in range(2)] for _ in range(m + 1)]
        for used in range(m + 1):
            for parity in (0, 1):
                cur = dp_hot[used][parity]
                if not cur:
                    continue
                for s, coeff in cur.items():
                    for k in range(0, m - used + 1):
                        mult = poly[k]
                        if mult == 0:
                            continue
                        # In a run of k hot components at temperature t:
                        # Right gets t extra points on each of Right's turns.
                        right_turns = (k + 1 - parity) // 2
                        delta = k * R + right_turns * t
                        nu = used + k
                        np = parity ^ (k & 1)
                        ns = s + delta
                        new[nu][np][ns] = (new[nu][np][ns] + coeff * mult) % MOD
        dp_hot = new

    # DP over integer-valued components (they simply add).
    dp_int = [defaultdict(int) for _ in range(m + 1)]
    dp_int[0][0] = 1
    for v, c in ints.items():
        poly = [0] * (m + 1)
        poly[0] = 1
        p = 1
        for k in range(1, m + 1):
            p = p * c % MOD
            poly[k] = p * invfact[k] % MOD

        new = [defaultdict(int) for _ in range(m + 1)]
        for used in range(m + 1):
            cur = dp_int[used]
            if not cur:
                continue
            for s, coeff in cur.items():
                for k in range(0, m - used + 1):
                    mult = poly[k]
                    if mult == 0:
                        continue
                    nu = used + k
                    ns = s + k * v
                    new[nu][ns] = (new[nu][ns] + coeff * mult) % MOD
        dp_int = new

    # Combine hot and integer parts, apply the win rule.
    # After all hot games are played, the position is a number (sum_value).
    # Right wins if sum_value > 0, or if sum_value == 0 and the number of hot games is odd
    # (equivalently parity == 1 at the transition to the purely numeric phase).
    multiset_count = 0
    for j in range(m + 1):
        for parity in (0, 1):
            for s_hot, ch in dp_hot[j][parity].items():
                for s_int, ci in dp_int[m - j].items():
                    total = s_hot + s_int
                    if total > 0 or (total == 0 and parity == 1):
                        multiset_count = (multiset_count + ch * ci) % MOD

    # Convert EGF-weighted multiset counts to ordered sequences.
    return multiset_count * fact[m] % MOD


def main() -> None:
    # Test values from the problem statement:
    assert solve(2, 4) == 7
    assert solve(3, 9) == 315319

    print(solve(8, 64) % MOD)


if __name__ == "__main__":
    main()
