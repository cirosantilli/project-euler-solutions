#!/usr/bin/env python3
"""Project Euler 882: Removing Bits

We model the position as a disjoint sum of independent components (each number in the multiset).
For each integer x >= 0 let g(x) be the combinatorial game value of the single-number position.
The initial position for n is then:
    G(n) = sum_{k=1..n} k * g(k)

Allowing Dr. Zero (Right) to "skip" a turn k times is equivalent to adding the game -k,
so Dr. Zero wins iff G(n) - k <= 0. The minimal such k is ceil(G(n)).

All component games here evaluate to dyadic rationals, so we compute them exactly using
integer arithmetic on a fixed power-of-two denominator.

No external libraries are used.
"""


def _simplest_between(a_scaled, b_scaled, max_exp):
    """Return the simplest dyadic strictly between the bounds.

    Values are represented as integers scaled by DEN = 2**max_exp.

    a_scaled is the lower bound (max of Left options), b_scaled is the upper bound (min of Right options).
    Use None for -infinity / +infinity.

    For this problem all values are non-negative, but this routine handles infinities defensively.
    """

    den = 1 << max_exp

    if a_scaled is None:  # (-inf, b)
        if b_scaled is None:
            return 0
        # simplest number strictly less than b is the greatest integer < b
        return ((b_scaled - 1) // den) * den

    if b_scaled is None:  # (a, +inf)
        # simplest number strictly greater than a is the smallest integer > a
        return ((a_scaled // den) + 1) * den

    a = a_scaled
    b = b_scaled
    if not (a < b):
        raise ValueError("No number lies strictly between bounds")

    # Search dyadics m / 2^k from simplest (coarsest) denominators upward.
    # At fixed k, choose the candidate closest to 0; with non-negative values this is just
    # the smallest one in the interval.
    for k in range(0, max_exp + 1):
        step = 1 << (max_exp - k)  # scale for denominator 2^k
        m_low = a // step + 1  # smallest m with m*step > a
        m_high = (b - 1) // step  # largest  m with m*step < b
        if m_low <= m_high:
            # For this problem the interval is always within non-negative numbers.
            return m_low * step

    # With k=max_exp, step=1, so we should always have found a candidate.
    raise RuntimeError("Failed to find a dyadic between two distinct bounds")


def compute_S(n):
    """Compute S(n) for the game described in Project Euler 882."""

    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        # n=0: empty position => S(0)=0 (not asked, but defined naturally)
        # n=1: [1] => value 1 => ceil(1)=1
        return n

    # A safe global denominator: for numbers < 2^L, the game values produced here only need
    # denominators up to 2^(L-1). Using 2^L is harmless and keeps all computations integral.
    max_exp = n.bit_length()  # for n=100000 this is 17
    den = 1 << max_exp

    g = [0] * (n + 1)  # g[x] is the scaled dyadic value of the single-number game for x

    total_scaled = 0

    checkpoints = {2: 2, 5: 17, 10: 64}

    for x in range(1, n + 1):
        # Generate option values by deleting one bit.
        left_opts = []
        right_opts = []

        bits = x.bit_length()
        for j in range(bits):  # j is the bit index from the LSB (0..bits-1)
            bit = (x >> j) & 1

            # Delete bit j from x.
            higher = x >> (j + 1)
            lower = x & ((1 << j) - 1)
            y = (higher << j) | lower

            if bit:
                left_opts.append(g[y])  # Dr. One removes a '1'
            else:
                right_opts.append(g[y])  # Dr. Zero removes a '0'

        # Canonical pruning for number games:
        # remove Left options that are >= some Right option (must be < value),
        # then remove Right options that are <= some remaining Left option (must be > value).
        changed = True
        while changed:
            changed = False
            if right_opts:
                min_r = min(right_opts)
                new_left = [v for v in left_opts if v < min_r]
                if len(new_left) != len(left_opts):
                    left_opts = new_left
                    changed = True
            if left_opts:
                max_l = max(left_opts)
                new_right = [v for v in right_opts if v > max_l]
                if len(new_right) != len(right_opts):
                    right_opts = new_right
                    changed = True

        max_l = max(left_opts) if left_opts else None
        min_r = min(right_opts) if right_opts else None

        g[x] = _simplest_between(max_l, min_r, max_exp)

        total_scaled += x * g[x]

        if x in checkpoints:
            got = (total_scaled + den - 1) // den
            assert got == checkpoints[x], f"S({x}) expected {checkpoints[x]}, got {got}"

    return (total_scaled + den - 1) // den


def main():
    n = 100_000

    # Asserts for test values explicitly given in the problem statement.
    assert compute_S(2) == 2
    assert compute_S(5) == 17
    assert compute_S(10) == 64

    print(compute_S(n))


if __name__ == "__main__":
    main()
