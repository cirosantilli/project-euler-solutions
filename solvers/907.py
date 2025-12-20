#!/usr/bin/env python3
"""
Project Euler 907: Stacking Cups

We have cups C1..Cn (in increasing size). Each cup can be oriented either:
- right-way-up (U)
- upside-down (D)

When stacking cup A directly on top of cup B, the following local rules apply:

1) "Nesting" between consecutive sizes (difference 1):
   - orientations are the same
   - if B is U, then A must be smaller (go down by 1)
   - if B is D, then A must be larger (go up by 1)

2) "Base-to-base" or "rim-to-rim" between sizes differing by 2:
   - orientations are opposite
   - the size can go up or down by 2

S(n) is the number of full towers (linear stacks) using all n cups.
We must compute S(10^7) mod 1_000_000_007.

Approach:
- Enumerate S(n) exactly for n <= 20 using memoised depth-first search
  (only for small n; this is fast because each step has <= 3 choices).
- Use a fixed linear recurrence of order 8 (validated against the small exact values).
- Evaluate S(10^7) with matrix exponentiation in O(log n) time.
"""

from functools import lru_cache
import sys


MOD = 1_000_000_007

# Linear recurrence (order 8), valid for n >= 10:
# S(n) = 2S(n-1) - 3S(n-2) + 5S(n-3) - 4S(n-4)
#        + 4S(n-5) - 3S(n-6) + S(n-7) - S(n-8)
REC_COEFF = (2, -3, 5, -4, 4, -3, 1, -1)


def count_towers_exact(n: int) -> int:
    """Exact S(n) for small n via memoised DFS over (used_mask, last, last_orientation).

    Cup labels are 0..n-1 (representing C1..Cn).
    orientation: 0 = U, 1 = D

    Next move rules from (last, ori):
      - to last±2 (if unused and in range), orientation flips
      - to last-1 if ori==U, orientation stays
      - to last+1 if ori==D, orientation stays
    """
    if n <= 0:
        return 0
    full = (1 << n) - 1

    @lru_cache(None)
    def dfs(mask: int, last: int, ori: int) -> int:
        if mask == full:
            return 1

        res = 0

        # Difference 2 moves (always allowed), orientation flips
        m2 = last - 2
        if m2 >= 0 and ((mask >> m2) & 1) == 0:
            res += dfs(mask | (1 << m2), m2, 1 - ori)

        p2 = last + 2
        if p2 < n and ((mask >> p2) & 1) == 0:
            res += dfs(mask | (1 << p2), p2, 1 - ori)

        # Difference 1 move (direction depends on current orientation), orientation stays
        nxt = last - 1 if ori == 0 else last + 1
        if 0 <= nxt < n and ((mask >> nxt) & 1) == 0:
            res += dfs(mask | (1 << nxt), nxt, ori)

        return res

    total = 0
    for i in range(n):
        bit = 1 << i
        total += dfs(bit, i, 0)  # start with i, U
        total += dfs(bit, i, 1)  # start with i, D
    return total


def mat_mul(A, B, mod: int):
    """Multiply two 8x8 matrices modulo mod."""
    k = 8
    res = [[0] * k for _ in range(k)]
    for i in range(k):
        Ai = A[i]
        ri = res[i]
        for m in range(k):
            a = Ai[m]
            if a:
                Bm = B[m]
                for j in range(k):
                    ri[j] = (ri[j] + a * Bm[j]) % mod
    return res


def mat_vec_mul(A, v, mod: int):
    """Multiply an 8x8 matrix by a length-8 vector modulo mod."""
    k = 8
    out = [0] * k
    for i in range(k):
        s = 0
        Ai = A[i]
        for j in range(k):
            s += Ai[j] * v[j]
        out[i] = s % mod
    return out


def apply_matrix_power_to_vector(M, exponent: int, vec, mod: int):
    """Compute (M^exponent) * vec via binary exponentiation."""
    A = [row[:] for row in M]
    v = vec[:]
    e = exponent
    while e > 0:
        if e & 1:
            v = mat_vec_mul(A, v, mod)
        e >>= 1
        if e:
            A = mat_mul(A, A, mod)
    return v


def solve(n: int) -> int:
    # Compute exact values up to 20 for asserts and to validate recurrence.
    small = [0] * 21  # 1-indexed: small[i] = S(i)
    for i in range(1, 21):
        small[i] = count_towers_exact(i)

    # Asserts required by the problem statement.
    assert small[4] == 12
    assert small[8] == 58
    assert small[20] == 5560

    # Validate recurrence on the range where we have exact data.
    for i in range(10, 21):
        rhs = 0
        for k, c in enumerate(REC_COEFF, start=1):
            rhs += c * small[i - k]
        assert rhs == small[i]

    if n <= 20:
        return small[n] % MOD

    # Companion matrix for the order-8 recurrence, using state:
    # [S(t), S(t-1), ..., S(t-7)]^T
    coeff_mod = [c % MOD for c in REC_COEFF]
    M = [[0] * 8 for _ in range(8)]
    M[0] = coeff_mod
    for i in range(1, 8):
        M[i][i - 1] = 1

    # Base state at t = 9 (so that next computed term is S(10), where recurrence is valid).
    base = [
        small[9] % MOD,
        small[8] % MOD,
        small[7] % MOD,
        small[6] % MOD,
        small[5] % MOD,
        small[4] % MOD,
        small[3] % MOD,
        small[2] % MOD,
    ]

    # We need S(n) = first component of M^(n-9) * base.
    vec_n = apply_matrix_power_to_vector(M, n - 9, base, MOD)
    return vec_n[0]


def main() -> None:
    sys.setrecursionlimit(10000)
    target = 10_000_000
    print(solve(target))


if __name__ == "__main__":
    main()
