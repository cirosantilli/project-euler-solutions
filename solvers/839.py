#!/usr/bin/env python3
"""Project Euler 839: Beans in Bowls

We have N bowls with initial counts S_0..S_{N-1}:
  S_0 = 290797
  S_n = S_{n-1}^2 mod 50515093

At each step, find the smallest index i such that a[i] > a[i+1],
then move one bean from bowl i to bowl i+1.
Let B(N) be the number of steps to reach a nondecreasing configuration.

This program computes B(10^7).

No external libraries are used (only Python standard library).
"""

from __future__ import annotations

from array import array


MOD = 50515093
S0 = 290797
TARGET_N = 10**7


def _new_u64_array() -> array:
    """Return an array suitable for holding up to ~5e14 without overflow."""
    # 'Q' is unsigned long long (typically 64-bit). Fallbacks are unlikely to be
    # needed on the judge, but provided for completeness.
    try:
        return array("Q")
    except ValueError:
        return array("L")


def compute_B(N: int) -> int:
    """Compute B(N) using a pooled-block (PAV-style) stabilization.

    Key ideas:
      * The stabilization is equivalent to finding the lexicographically largest
        nondecreasing integer sequence reachable by only moving beans to the right.
      * Represent the final sequence as consecutive blocks. A block of length L
        with total sum T is as flat as possible:
            base = T // L
            rem  = T % L
        giving values: base (L-rem times), then base+1 (rem times).
      * Adjacent blocks are valid iff last(left) <= first(right), i.e.:
            ceil(T_left/L_left) <= floor(T_right/L_right)
        If violated, merge the two blocks and re-evaluate (amortized O(N)).
      * Each move increases the potential sum(i * a[i]) by exactly 1, so:
            B(N) = potential(final) - potential(initial)
    """
    if N <= 1:
        return 0

    # Stacks of block lengths and block sums.
    lens = array("I")
    sums = _new_u64_array()

    s = S0
    init_potential = 0  # Python int (can grow beyond 64-bit)

    append_len = lens.append
    append_sum = sums.append

    for i in range(N):
        v = s
        init_potential += i * v

        append_len(1)
        append_sum(v)

        # Merge while last(left) > first(right).
        while len(lens) >= 2:
            l_left = lens[-2]
            t_left = sums[-2]
            l_right = lens[-1]
            t_right = sums[-1]

            # last(left) = ceil(t_left / l_left)
            # first(right) = floor(t_right / l_right)
            if (t_left + l_left - 1) // l_left <= t_right // l_right:
                break

            # merge
            lens[-2] = l_left + l_right
            sums[-2] = t_left + t_right
            lens.pop()
            sums.pop()

        s = (v * v) % MOD

    # Compute potential(final) from the blocks, without expanding to length N.
    pos = 0
    final_potential = 0

    for L, T in zip(lens, sums):
        # Block distribution: base (L-rem times), then base+1 (rem times).
        base = T // L
        rem = T - base * L  # same as T % L

        # Sum of indices in [pos, pos+L-1]
        sum_all = L * (2 * pos + L - 1) // 2
        final_potential += base * sum_all

        if rem:
            # Extra +1 on the last rem indices.
            start_extra = pos + L - rem
            sum_extra = rem * (2 * start_extra + rem - 1) // 2
            final_potential += sum_extra

        pos += L

    return final_potential - init_potential


def main() -> None:
    # Tests from the problem statement.
    assert compute_B(5) == 0
    assert compute_B(6) == 14263289
    assert compute_B(100) == 3284417556

    print(compute_B(TARGET_N))


if __name__ == "__main__":
    main()
