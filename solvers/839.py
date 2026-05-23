#!/usr/bin/env python
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

MOD = 50515093
S0 = 290797
TARGET_N = 10**7


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
      * Each rightward move decreases exactly the prefix sums it crosses, so:
            B(N) = sum(prefix(initial)) - sum(prefix(final))
    """
    if N <= 1:
        return 0

    # Stacks of block lengths and block sums.
    lens: list[int] = []
    sums: list[int] = []

    s = S0
    prefix = 0
    initial_prefix_total = 0

    append_len = lens.append
    append_sum = sums.append
    pop_len = lens.pop
    pop_sum = sums.pop
    size = 0

    for _ in range(N):
        v = s
        prefix += v
        initial_prefix_total += prefix

        append_len(1)
        append_sum(v)
        size += 1

        # Merge while last(left) > first(right).
        while size >= 2:
            last = size - 1
            l_left = lens[last - 1]
            t_left = sums[last - 1]
            l_right = lens[last]
            t_right = sums[last]

            # last(left) = ceil(t_left / l_left)
            # first(right) = floor(t_right / l_right)
            if (t_left + l_left - 1) // l_left <= t_right // l_right:
                break

            # merge
            lens[last - 1] = l_left + l_right
            sums[last - 1] = t_left + t_right
            pop_len()
            pop_sum()
            size -= 1

        s = (v * v) % MOD

    # Compute sum(prefix(final)) from the blocks, without expanding to length N.
    running_prefix = 0
    final_prefix_total = 0

    for index in range(size):
        L = lens[index]
        T = sums[index]
        # Block distribution: base (L-rem times), then base+1 (rem times).
        base, rem = divmod(T, L)
        low_count = L - rem

        final_prefix_total += (
            low_count * running_prefix + base * low_count * (low_count + 1) // 2
        )

        if rem:
            after_low = running_prefix + low_count * base
            final_prefix_total += rem * after_low + (base + 1) * rem * (rem + 1) // 2

        running_prefix += T

    return initial_prefix_total - final_prefix_total


def main() -> None:
    # Tests from the problem statement.
    assert compute_B(5) == 0
    assert compute_B(6) == 14263289
    assert compute_B(100) == 3284417556

    print(compute_B(TARGET_N))


if __name__ == "__main__":
    main()
