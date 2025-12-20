#!/usr/bin/env python3
"""
Project Euler 944 - Sum of Elevisors

Run:
  python3 main.py

Optional:
  python3 main.py <n>
  python3 main.py <n> <mod>

No external libraries are used.
"""

import sys
import math


def sev(E):
    """
    Sum of elevisors of a set E.
    x in E is an elevisor if it divides another (different) element of E.
    """
    a = list(E)
    s = 0
    for i, x in enumerate(a):
        for j, y in enumerate(a):
            if i != j and y % x == 0:
                s += x
                break
    return s


def _sum_range_mod(l, r, mod):
    """sum_{k=l..r} k (mod mod), with l<=r, using modular-safe halving."""
    cnt = r - l + 1
    a = l + r
    if a & 1 == 0:
        a //= 2
    else:
        cnt //= 2
    return (a % mod) * (cnt % mod) % mod


def S(n, mod):
    """
    Compute S(n) = sum_{E subset of {1..n}} sev(E)  (mod mod).

    Derivation:
      For each x, it contributes x to sev(E) iff:
        - x is chosen, and
        - at least one other multiple of x (<=n) is chosen.
      Count of such subsets:
        2^(n-1) - 2^(n - floor(n/x)).
      Hence:
        S(n) = sum_{x=1..n} x * (2^(n-1) - 2^(n - floor(n/x))).

    We evaluate the sum in ~O(sqrt(n)) modular operations with:
      - a sqrt split,
      - arithmetic progression sums,
      - and incremental updates of 2^{-floor(n/x)}.
    """
    if n <= 0:
        return 0

    # mod is odd in the problem; still compute inverse robustly.
    inv2 = pow(2, -1, mod)
    two = 2 % mod

    pow2_n = pow(2, n, mod)  # 2^n (mod mod)
    pow2_n_minus1 = (pow2_n * inv2) % mod  # 2^(n-1) (mod mod)

    # n(n+1)/2 mod, avoiding huge intermediate where possible
    if n & 1 == 0:
        half_nnp1 = ((n // 2) % mod) * ((n + 1) % mod) % mod
    else:
        half_nnp1 = (n % mod) * (((n + 1) // 2) % mod) % mod

    first_term = pow2_n_minus1 * half_nnp1 % mod  # 2^(n-1) * sum_{x=1..n} x

    # We need U = sum_{x=1..n} x * 2^{-floor(n/x)}  (mod mod),
    # then second_term = 2^n * U (mod mod).
    B = math.isqrt(n)

    # For small-x iteration we update inv2^{floor(n/x)} via deltas:
    # if q decreases by delta, inv2^{q-delta} = inv2^q * 2^delta.
    # Most deltas are small; precompute 2^d for d <= L.
    L = 200_000
    pow2_small = [1] * (L + 1)
    for i in range(1, L + 1):
        pow2_small[i] = (pow2_small[i - 1] * 2) % mod

    # Part 1: x = 1..B
    U_small = 0
    q = n
    a = pow(inv2, q, mod)  # a = inv2^q
    for x in range(1, B + 1):
        U_small = (U_small + x * a) % mod
        if x == B:
            break
        next_q = n // (x + 1)
        delta = q - next_q
        if delta <= L:
            a = (a * pow2_small[delta]) % mod
        else:
            a = (a * pow(2, delta, mod)) % mod
        q = next_q

    # Part 2: x = B+1..n  (equivalently q = floor(n/x) in 1..B)
    # Iterate q downward and reuse the identity:
    #   interval(q) = (n/(q+1), n/q]  => l = floor(n/(q+1))+1, r=floor(n/q)
    # We can get l from the previous r: l_q = r_{q+1} + 1.
    U_large = 0
    prev_r = n // (B + 1)  # r_{B+1}
    weight = pow(inv2, B, mod)  # inv2^B
    for q in range(B, 0, -1):
        r = n // q
        l = prev_r + 1
        prev_r = r

        # Only count x > B (x <= B already in U_small)
        if r <= B:
            weight = (weight * two) % mod
            continue
        if l <= B:
            l = B + 1

        if l <= r:
            sum_x = _sum_range_mod(l, r, mod)
            U_large = (U_large + sum_x * weight) % mod

        # Move to q-1: inv2^(q-1) = inv2^q * inv2^{-1} = inv2^q * 2
        weight = (weight * two) % mod

    U = (U_small + U_large) % mod
    second_term = (pow2_n * U) % mod

    return (first_term - second_term) % mod


def main():
    # Required asserts from the problem statement examples.
    assert sev({1, 2, 5, 6}) == 3
    assert S(10, 1234567891) == 4927

    n = 10**14
    mod = 1234567891
    if len(sys.argv) >= 2:
        n = int(sys.argv[1])
    if len(sys.argv) >= 3:
        mod = int(sys.argv[2])

    print(S(n, mod))


if __name__ == "__main__":
    main()
