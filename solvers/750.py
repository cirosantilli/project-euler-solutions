#!/usr/bin/env python3
"""
Project Euler 750: Optimal Card Stacking

We start with N cards placed in positions 1..N. The card in position n is 3^n mod (N+1).
A move consists of dragging one stack onto another stack with horizontal cost |pos_from - pos_to|.
The move is allowed only if the resulting stack is a consecutive sequence.

Interpretation consistent with given examples:
- A stack is in sequence top-to-bottom increasing (e.g. [1,2,3,...]).
- Dragging stack A onto stack B places A on top of B.
- Therefore allowed only when max(A)+1 == min(B).
- When merging [l..k] onto [k+1..r], the merged stack becomes [l..r] and stays at position of r.

This yields an interval DP:
dp[l][r] = min_{k in [l..r-1]} dp[l][k] + dp[k+1][r] + |pos[k] - pos[r]|.
Answer is dp[1][N].
"""


def generate_positions(N: int):
    """
    Build pos[x] = position index where card label x appears.
    Returns None if the arrangement is not a permutation of 1..N (i.e. G(N) undefined).
    """
    mod = N + 1
    pos = [0] * (N + 1)

    x = 1
    for i in range(1, N + 1):
        x = (x * 3) % mod
        if x == 0 or x > N or pos[x] != 0:
            return None
        pos[x] = i

    # Ensure all labels 1..N appear
    for v in range(1, N + 1):
        if pos[v] == 0:
            return None

    return pos


def G(N: int) -> int:
    """
    Compute G(N) using O(N^3) interval DP with a small constant optimization.
    """
    pos = generate_positions(N)
    if pos is None:
        raise ValueError(f"G({N}) is not defined (sequence is not a permutation).")

    # dp[l][r] minimal cost to build stack [l..r] (top-to-bottom increasing)
    dp = [[0] * (N + 1) for _ in range(N + 1)]
    INF = 10**18
    pos_local = pos

    # Process by increasing right endpoint r
    # For fixed r, we compute dp[l][r] for l=r-1 down to 1.
    # Maintain a local column cache col[i] = dp[i][r] as it becomes known.
    for r in range(2, N + 1):
        pr = pos_local[r]
        col = [0] * (r + 2)  # col[r]=0 already, also safe access col[k+1]

        for l in range(r - 1, 0, -1):
            dp_l = dp[l]
            best = INF

            # try split at k: [l..k] onto [k+1..r]
            for k in range(l, r):
                d = pos_local[k] - pr
                if d < 0:
                    d = -d
                val = dp_l[k] + col[k + 1] + d
                if val < best:
                    best = val

            dp_l[r] = best
            col[l] = best

    return dp[1][N]


def main():
    # Asserts required by the statement
    assert G(6) == 8
    assert G(16) == 47

    # Print the required value (not asserted, not hardcoded)
    print(G(976))


if __name__ == "__main__":
    main()
