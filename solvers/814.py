#!/usr/bin/env python3
"""
Project Euler 814 - Mezzo-forte

Count assignments of "where each person looks" among {left, right, opposite} for 4n people
on a circle such that exactly half the people scream.

A person screams iff they are in a mutual-looking pair (a 2-cycle). Therefore, "exactly half
scream" means exactly n mutual pairs among 4n people.

We compute S(n) modulo 998244353 with a width-2 transfer DP (state compression) over
2n columns in a 2 x (2n) "twisted ladder" representation of the circle + opposite chords.
"""

MOD = 998244353


def _swap2bits(x: int) -> int:
    """Swap two low bits: b0b1 -> b1b0."""
    return ((x & 1) << 1) | ((x >> 1) & 1)


def S_mod(n: int, mod: int = MOD) -> int:
    """
    Return S(n) modulo `mod`.

    DP idea:
      - Split the 4n-cycle into 2n columns, each column has a "top" and "bottom" node
        which are opposite (diametrical).
      - Horizontal adjacency is along rows, except the wrap-around swaps rows ("twist"),
        matching the original single cycle order.
      - A mutual pair occurs:
          * on the left edge of a node if (incoming_from_left == 1) and (node chooses Left)
          * on the vertical edge in a column iff both choose Opposite
        Horizontal mutual pairs are counted exactly once: at the right endpoint (when it
        chooses Left and sees an incoming Right from its left neighbor).
      - State is a 2-bit mask describing whether each node in the current column has
        an incoming "Right" from the previous column (top bit0, bottom bit1).

    We track a polynomial in x whose exponent is the number of mutual pairs.
    We only need the coefficient of x^n, so we truncate degrees > n.
    """
    if n < 1:
        raise ValueError("n must be a positive integer")

    target = n
    cols = 2 * n  # number of columns in the 2-row representation

    # Precomputed transition groups for each incoming mask.
    #
    # For a fixed incoming mask (in_top, in_bottom), each column's two people independently
    # choose among {L, R, V}. There are 9 choices total. Each choice implies:
    #   - out_mask: which of (top,bottom) chose R (so the next column sees incoming rights)
    #   - delta: how many mutual pairs are created "now" (0..2)
    #
    # We group equal (out_mask, delta) combinations with multiplicity count in {1,2,3}.
    GROUPS = [
        # mask 0: in_top=0, in_bottom=0
        [(0, 0, 3), (2, 0, 2), (1, 0, 2), (3, 0, 1), (0, 1, 1)],
        # mask 1: in_top=1, in_bottom=0
        [(0, 1, 3), (2, 1, 1), (1, 0, 2), (3, 0, 1), (0, 0, 1), (2, 0, 1)],
        # mask 2: in_top=0, in_bottom=1
        [(0, 1, 3), (2, 0, 2), (0, 0, 1), (1, 1, 1), (3, 0, 1), (1, 0, 1)],
        # mask 3: in_top=1, in_bottom=1
        [(0, 2, 1), (2, 1, 1), (0, 1, 3), (1, 1, 1), (3, 0, 1), (1, 0, 1), (2, 0, 1)],
    ]

    # Same groups, but with the "twist" applied to the outgoing mask on the final step.
    GROUPS_LAST = []
    for mask_in in range(4):
        gl = []
        for out_mask, delta, cnt in GROUPS[mask_in]:
            gl.append((_swap2bits(out_mask), delta, cnt))
        GROUPS_LAST.append(gl)

    # Sum over all possible initial incoming masks; enforce cyclic consistency by
    # requiring the final incoming mask equals the initial one.
    total = 0

    for init_in in range(4):
        # dp[mask][k] = number of ways so far with current incoming mask `mask`
        # and exactly k mutual pairs counted so far.
        dp = [[0] * (target + 1) for _ in range(4)]
        dp[init_in][0] = 1

        # Process columns 0 .. cols-2 (normal steps; right edges keep row identity)
        for _ in range(cols - 1):
            dp_next = [[0] * (target + 1) for _ in range(4)]
            for mask_in in range(4):
                arr = dp[mask_in]
                for out_mask, delta, cnt in GROUPS[mask_in]:
                    dest = dp_next[out_mask]

                    if cnt == 1:
                        if delta == 0:
                            for k in range(target + 1):
                                x = dest[k] + arr[k]
                                if x >= mod:
                                    x -= mod
                                dest[k] = x
                        elif delta == 1:
                            for k in range(target):
                                idx = k + 1
                                x = dest[idx] + arr[k]
                                if x >= mod:
                                    x -= mod
                                dest[idx] = x
                        else:  # delta == 2
                            for k in range(target - 1):
                                idx = k + 2
                                x = dest[idx] + arr[k]
                                if x >= mod:
                                    x -= mod
                                dest[idx] = x

                    elif cnt == 2:
                        if delta == 0:
                            for k in range(target + 1):
                                add = arr[k] * 2
                                if add >= mod:
                                    add -= mod
                                x = dest[k] + add
                                if x >= mod:
                                    x -= mod
                                dest[k] = x
                        elif delta == 1:
                            for k in range(target):
                                add = arr[k] * 2
                                if add >= mod:
                                    add -= mod
                                idx = k + 1
                                x = dest[idx] + add
                                if x >= mod:
                                    x -= mod
                                dest[idx] = x
                        else:  # delta == 2
                            for k in range(target - 1):
                                add = arr[k] * 2
                                if add >= mod:
                                    add -= mod
                                idx = k + 2
                                x = dest[idx] + add
                                if x >= mod:
                                    x -= mod
                                dest[idx] = x

                    else:  # cnt == 3
                        if delta == 0:
                            for k in range(target + 1):
                                add = arr[k] * 3
                                if add >= mod:
                                    add -= mod
                                if add >= mod:
                                    add -= mod
                                x = dest[k] + add
                                if x >= mod:
                                    x -= mod
                                dest[k] = x
                        elif delta == 1:
                            for k in range(target):
                                add = arr[k] * 3
                                if add >= mod:
                                    add -= mod
                                if add >= mod:
                                    add -= mod
                                idx = k + 1
                                x = dest[idx] + add
                                if x >= mod:
                                    x -= mod
                                dest[idx] = x
                        else:  # delta == 2
                            for k in range(target - 1):
                                add = arr[k] * 3
                                if add >= mod:
                                    add -= mod
                                if add >= mod:
                                    add -= mod
                                idx = k + 2
                                x = dest[idx] + add
                                if x >= mod:
                                    x -= mod
                                dest[idx] = x

            dp = dp_next

        # Final column (wrap-around step has row swap)
        dp_next = [[0] * (target + 1) for _ in range(4)]
        for mask_in in range(4):
            arr = dp[mask_in]
            for next_in, delta, cnt in GROUPS_LAST[mask_in]:
                dest = dp_next[next_in]

                if cnt == 1:
                    if delta == 0:
                        for k in range(target + 1):
                            x = dest[k] + arr[k]
                            if x >= mod:
                                x -= mod
                            dest[k] = x
                    elif delta == 1:
                        for k in range(target):
                            idx = k + 1
                            x = dest[idx] + arr[k]
                            if x >= mod:
                                x -= mod
                            dest[idx] = x
                    else:  # delta == 2
                        for k in range(target - 1):
                            idx = k + 2
                            x = dest[idx] + arr[k]
                            if x >= mod:
                                x -= mod
                            dest[idx] = x

                elif cnt == 2:
                    if delta == 0:
                        for k in range(target + 1):
                            add = arr[k] * 2
                            if add >= mod:
                                add -= mod
                            x = dest[k] + add
                            if x >= mod:
                                x -= mod
                            dest[k] = x
                    elif delta == 1:
                        for k in range(target):
                            add = arr[k] * 2
                            if add >= mod:
                                add -= mod
                            idx = k + 1
                            x = dest[idx] + add
                            if x >= mod:
                                x -= mod
                            dest[idx] = x
                    else:  # delta == 2
                        for k in range(target - 1):
                            add = arr[k] * 2
                            if add >= mod:
                                add -= mod
                            idx = k + 2
                            x = dest[idx] + add
                            if x >= mod:
                                x -= mod
                            dest[idx] = x

                else:  # cnt == 3
                    if delta == 0:
                        for k in range(target + 1):
                            add = arr[k] * 3
                            if add >= mod:
                                add -= mod
                            if add >= mod:
                                add -= mod
                            x = dest[k] + add
                            if x >= mod:
                                x -= mod
                            dest[k] = x
                    elif delta == 1:
                        for k in range(target):
                            add = arr[k] * 3
                            if add >= mod:
                                add -= mod
                            if add >= mod:
                                add -= mod
                            idx = k + 1
                            x = dest[idx] + add
                            if x >= mod:
                                x -= mod
                            dest[idx] = x
                    else:  # delta == 2
                        for k in range(target - 1):
                            add = arr[k] * 3
                            if add >= mod:
                                add -= mod
                            if add >= mod:
                                add -= mod
                            idx = k + 2
                            x = dest[idx] + add
                            if x >= mod:
                                x -= mod
                            dest[idx] = x

        dp = dp_next

        # Enforce cycle closure: final incoming mask equals initial incoming mask.
        total += dp[init_in][target]
        total %= mod

    return total


def main() -> None:
    # Test values from the problem statement.
    assert S_mod(1) == 48
    assert S_mod(10) == 420121075

    # Problem asks for S(10^3) mod 998244353.
    print(S_mod(1000))


if __name__ == "__main__":
    main()
