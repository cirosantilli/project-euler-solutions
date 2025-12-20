#!/usr/bin/env python3
"""Project Euler 865 - Triplicate Numbers

A "triplicate" digit string can be reduced to the empty string by repeatedly
removing any occurrence of three consecutive equal digits.

Let T(n) be the count of positive integers less than 10^n that are triplicate.
Equivalently: the count of digit strings of length 1..n with no leading zero
that are triplicate.

We need T(10^4) modulo 998244353.

The problem statement provides:
  T(6)  = 261
  T(30) = 5576195181577716

These are asserted below.

No external libraries are used.
"""

MOD = 998244353
K = 10


def _compute_T_via_dp(n: int, mod: int | None) -> int:
    """Compute T(n).

    If mod is None, returns the exact integer (int can grow big; intended only
    for small n used in assertions).

    If mod is an int, returns T(n) % mod.

    The DP is performed only for lengths divisible by 3 (necessary condition).
    We work with m = length/3.
    """

    if n <= 0:
        return 0

    mmax = n // 3  # only lengths 3,6,9,... contribute
    if mmax == 0:
        return 0

    # dp0[m]: triplicate strings of length 3*m over digits 0-9 (no leading rule)
    # dp1[m]: triplicate strings of length 3*m with first digit != 0
    # f[m]:   auxiliary count used in the decomposition (see README)
    # mul[m]: convolution of f with itself: sum_{a=1..m-1} f[a]*f[m-a]

    dp0 = [0] * (mmax + 1)
    dp1 = [0] * (mmax + 1)
    f = [0] * (mmax + 1)
    mul = [0] * (mmax + 1)
    pref = [0] * (mmax + 1)

    # Base case: length 3.
    dp0[1] = K
    dp1[1] = K - 1
    f[1] = K - 1
    pref[1] = dp1[1]

    # Constants used repeatedly.
    k = K
    km1 = K - 1
    twok = 2 * K
    twokm1 = 2 * (K - 1)

    # A soft threshold to periodically reduce modulo and keep integers small.
    # (Only used in modular mode.)
    reduce_every = 64

    for m in range(2, mmax + 1):
        if mod is None:
            f_m = km1 * f[m - 1]
            dp0_m = k * dp0[m - 1]
            dp1_m = k * dp1[m - 1]
        else:
            f_m = (km1 * f[m - 1]) % mod
            dp0_m = (k * dp0[m - 1]) % mod
            dp1_m = (k * dp1[m - 1]) % mod

        # Choose the size s (in triplets) of the suffix that contains the three
        # distinguished digits responsible for the final deletion step.
        for s in range(2, m + 1):
            p = m - s  # prefix length in triplets
            x = f[s - 1]

            if p == 0:
                # The distinguished digit is at the first position of the whole
                # string; leading-zero restriction affects dp1.
                if mod is None:
                    f_m += twokm1 * x
                    dp0_m += twok * x
                    dp1_m += twokm1 * x
                    if s >= 3:
                        y = mul[s - 1]
                        f_m += km1 * y
                        dp0_m += k * y
                        dp1_m += km1 * y
                else:
                    f_m += twokm1 * x
                    dp0_m += twok * x
                    dp1_m += twokm1 * x
                    if s >= 3:
                        y = mul[s - 1]
                        f_m += km1 * y
                        dp0_m += k * y
                        dp1_m += km1 * y
            else:
                fp = f[p]
                d0p = dp0[p]
                d1p = dp1[p]

                if mod is None:
                    f_m += twokm1 * x * fp
                    dp0_m += twok * x * d0p
                    dp1_m += twok * x * d1p
                    if s >= 3:
                        y = mul[s - 1]
                        f_m += km1 * y * fp
                        dp0_m += k * y * d0p
                        dp1_m += k * y * d1p
                else:
                    f_m += twokm1 * x * fp
                    dp0_m += twok * x * d0p
                    dp1_m += twok * x * d1p
                    if s >= 3:
                        y = mul[s - 1]
                        f_m += km1 * y * fp
                        dp0_m += k * y * d0p
                        dp1_m += k * y * d1p

            if mod is not None and (s & (reduce_every - 1)) == 0:
                f_m %= mod
                dp0_m %= mod
                dp1_m %= mod

        if mod is not None:
            f_m %= mod
            dp0_m %= mod
            dp1_m %= mod

        f[m] = f_m
        dp0[m] = dp0_m
        dp1[m] = dp1_m

        # Update mul[m] = sum_{a=1..m-1} f[a] * f[m-a]
        if mod is None:
            acc = 0
            for a in range(1, m):
                acc += f[a] * f[m - a]
            mul[m] = acc
        else:
            acc = 0
            for a in range(1, m):
                acc += f[a] * f[m - a]
                if (a & (reduce_every - 1)) == 0:
                    acc %= mod
            mul[m] = acc % mod

        if mod is None:
            pref[m] = pref[m - 1] + dp1[m]
        else:
            pref[m] = (pref[m - 1] + dp1[m]) % mod

    return pref[mmax] if mod is None else (pref[mmax] % mod)


def T_exact(n: int) -> int:
    """Exact T(n) for small n (used for the statement's test values)."""
    return _compute_T_via_dp(n, mod=None)


def T_mod(n: int, mod: int = MOD) -> int:
    """T(n) modulo mod."""
    return _compute_T_via_dp(n, mod=mod)


def main() -> None:
    # Test values from the problem statement.
    assert T_exact(6) == 261
    assert T_exact(30) == 5576195181577716

    print(T_mod(10_000))


if __name__ == "__main__":
    main()
