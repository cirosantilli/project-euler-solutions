#!/usr/bin/env python3
"""
Project Euler 925
-----------------
Let B(n) be the smallest number larger than n that can be formed by rearranging
digits of n, or 0 if no such number exists. Define

    T(N) = sum_{n=1..N} B(n^2)

Compute T(10^16) modulo 1_000_000_007.

No external libraries are used.
"""

MOD = 1_000_000_007


def next_permutation(digs):
    """
    In-place next lexicographic permutation of a list of digits (msd -> lsd).
    Returns True if it exists, False if the sequence is non-increasing.
    """
    i = len(digs) - 2
    while i >= 0 and digs[i] >= digs[i + 1]:
        i -= 1
    if i < 0:
        return False
    j = len(digs) - 1
    while digs[j] <= digs[i]:
        j -= 1
    digs[i], digs[j] = digs[j], digs[i]
    digs[i + 1 :] = reversed(digs[i + 1 :])
    return True


def digits_to_int(digs):
    x = 0
    for d in digs:
        x = x * 10 + d
    return x


def B(n):
    """Compute B(n) exactly (as a Python int)."""
    digs = [ord(c) - 48 for c in str(n)]
    if not next_permutation(digs):
        return 0
    return digits_to_int(digs)


def T_bruteforce(N):
    """Brute force T(N) for small N (used only for asserted examples)."""
    s = 0
    for n in range(1, N + 1):
        s += B(n * n)
    return s


def sum_squares_mod(N):
    """Sum_{n=1..N} n^2 (mod MOD)."""
    if N <= 0:
        return 0
    n = N % MOD
    n1 = (N + 1) % MOD
    n2 = (2 * N + 1) % MOD
    inv6 = pow(6, MOD - 2, MOD)
    return (n * n1 % MOD) * (n2 * inv6 % MOD) % MOD


def delta_from_suffix_digits_low(digits_low, s_val):
    """
    Given the last t digits of a number in digits_low order (lsd -> msd),
    and the numeric value of those digits (s_val), compute:

        delta = B_suffix - suffix

    where B_suffix is the next-permutation of the *t-digit sequence*.
    """
    msd = digits_low[::-1]
    ok = next_permutation(msd)
    assert ok  # only called when an ascent exists in the suffix
    new_s = digits_to_int(msd)
    return new_s - s_val


def sum_delta_for_length(L, pow10, pow10_mod):
    """
    Sum over all L-digit n:  (B(n^2) - n^2)  modulo MOD.

    Suffix pruning:
      - Build n from least significant digit upward.
      - Track a suffix v with length c.
      - Track tz = number of confirmed trailing zeros in v.
      - Use t = c + tz digits of v^2 (mod 10^t). When those t digits are not
        non-increasing (msd->lsd), the next-permutation pivot lies inside the
        suffix, so the delta is identical for all extensions.
    """
    total = 0
    mod = MOD

    start_digits = range(1, 10) if L == 1 else range(10)
    stack = []
    for d0 in start_digits:
        tz0 = 1 if d0 == 0 else 0
        stack.append((1, d0, tz0))

    while stack:
        c, v, tz = stack.pop()

        t = c + tz
        s = (v * v) % pow10[t]

        # Extract exactly t digits (lsd -> msd) and test if they are non-decreasing
        # in that order (equivalently, non-increasing in msd->lsd order).
        digits_low = []
        tmp = s
        prev = -1
        nondecreasing = True
        for _ in range(t):
            tmp, d = divmod(tmp, 10)
            if prev > d:
                nondecreasing = False
            prev = d
            digits_low.append(d)

        if not nondecreasing:
            delta = delta_from_suffix_digits_low(digits_low, s)
            if c < L:
                # Remaining digits: L-c, with the most significant digit forced nonzero.
                count_mod = (9 * pow10_mod[L - c - 1]) % mod
                total = (total + (delta % mod) * count_mod) % mod
            else:
                total = (total + delta) % mod
            continue

        if c == L:
            x = v * v
            delta_full = B(x) - x
            total = (total + delta_full) % mod
            continue

        next_is_msd = c + 1 == L
        digit_start = 1 if next_is_msd else 0

        # Push children in reverse digit order (depth-first, small stack).
        for d in range(9, digit_start - 1, -1):
            nv = v + d * pow10[c]
            ntz = tz
            # trailing-zero count grows only while all confirmed digits are zeros
            if tz == c and d == 0:
                ntz = c + 1
            stack.append((c + 1, nv, ntz))

    return total % mod


def solve():
    # Statement examples / test values
    assert B(245) == 254
    assert B(542) == 0
    assert T_bruteforce(10) == 270
    assert T_bruteforce(100) == 335316

    # T(10^16) includes n=10^16, but (10^16)^2 = 10^32 has digits "1" followed by
    # zeros only, so no larger rearrangement exists and B((10^16)^2)=0.
    # Therefore T(10^16) = T(10^16 - 1).
    N = 10**16 - 1

    # Powers of 10 as integers (for modulus 10^t in suffix checks)
    pow10 = [1]
    for _ in range(1, 33):
        pow10.append(pow10[-1] * 10)

    # Powers of 10 modulo MOD (for counting how many completions remain)
    pow10_mod = [1]
    for _ in range(1, 17):
        pow10_mod.append((pow10_mod[-1] * 10) % MOD)

    sum_delta = 0
    for L in range(1, 17):
        sum_delta = (sum_delta + sum_delta_for_length(L, pow10, pow10_mod)) % MOD

    ans = (sum_squares_mod(N) + sum_delta) % MOD
    print(ans)


if __name__ == "__main__":
    solve()
