#!/usr/bin/env python3
"""
Project Euler 775 - Saving Paper

We model an arrangement of n unit cubes as a polycube on the integer lattice.
If the wrapping paper touches the cubes everywhere (no "bridging"), then the
required amount of paper equals the surface area of the polycube.

Let s(n) be the minimum possible surface area among all valid polycubes with n cubes.
Then g(n) (maximum paper saved) is:

    g(n) = 6n - s(n)

We need:
    G(N) = sum_{n=1..N} g(n)  (mod 1_000_000_007)

This implementation uses a known structural characterization of surface-minimizing
polycubes: between k^3 and (k+1)^3 cubes, one starts from a k×k×k cube and adds
three orthogonal layers of sizes k^2, k(k+1), (k+1)^2. Within each layer, only
O(sqrt(m)) of the first m added cubes increase the surface area by 2; the rest
increase it by 0 (except the first cube of each layer, which increases it by 4).
That yields a closed-form for s(n) and for summing s(n) over huge ranges.

No external libraries are used.
"""

import math

MOD = 1_000_000_007


def icbrt_floor(x: int) -> int:
    """Return floor(cuberoot(x)) for x >= 0 using integer binary search."""
    if x < 0:
        raise ValueError("x must be non-negative")
    lo, hi = 0, 1
    while hi * hi * hi <= x:
        hi *= 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if mid * mid * mid <= x:
            lo = mid
        else:
            hi = mid
    return lo


def c_count(m: int) -> int:
    """
    Count of 'turn' positions in the first m steps of the (square) spiral layer,
    i.e. number of j >= 2 with floor(j^2/4) + 1 <= m.

    Closed form:
        c(m) = max(0, isqrt(4m-1) - 1)
    """
    if m <= 1:
        return 0
    return math.isqrt(4 * m - 1) - 1


def c_prefix_sum(t: int) -> int:
    """
    F(t) = sum_{m=1..t} c_count(m)

    Using the block structure of floor(sqrt(4m-1)):
    For a >= 1, on m in (a^2, (a+1)^2] the values form two constant runs,
    giving a total contribution of 4a^2 + a for that whole interval.

    This yields an O(1) prefix-sum formula.
    """
    if t <= 1:
        return 0
    a = math.isqrt(t)  # a^2 <= t < (a+1)^2

    # Sum of full intervals (i^2, (i+1)^2] for i=1..a-1:
    # sum_{i=1..a-1} (4i^2 + i) = (a-1)*a*(8a-1)/6
    base = (a - 1) * a * (8 * a - 1) // 6

    # Partial within (a^2, t]
    sq = a * a
    if t == sq:
        return base

    # Even part: m in [a^2+1, a^2+a] where c(m) = 2a-1
    end_even = min(t, sq + a)
    cnt_even = end_even - (sq + 1) + 1
    partial = cnt_even * (2 * a - 1)

    # Odd part: m in [a^2+a+1, (a+1)^2] where c(m) = 2a
    if t > sq + a:
        cnt_odd = t - (sq + a + 1) + 1
        partial += cnt_odd * (2 * a)

    return base + partial


def smin(n: int) -> int:
    """Minimum surface area for n unit cubes."""
    if n <= 0:
        raise ValueError("n must be positive")
    if n == 1:
        return 6

    k = icbrt_floor(n - 1)  # so k^3 < n <= (k+1)^3
    kz = n - k * k * k

    k2 = k * k
    cap2 = k * (k + 1)

    # One-time +4 when starting each of the 3 orthogonal layers
    if kz > k2 + cap2:
        bv = 12
    elif kz > k2:
        bv = 8
    else:
        bv = 4

    pz = min(kz, k2)
    qz = min(max(kz - pz, 0), cap2)
    rz = min(max(kz - pz - qz, 0), (k + 1) * (k + 1))

    cv = 2 * (c_count(pz) + c_count(qz) + c_count(rz))
    return 6 * k2 + bv + cv


def g(n: int) -> int:
    """Maximum paper saved for n cubes."""
    return 6 * n - smin(n)


def sum_smin_mod(N: int, mod: int = MOD) -> int:
    """
    Compute S(N) = sum_{n=1..N} smin(n) modulo mod.

    Uses block decomposition by k = floor((n-1)^(1/3)).
    """
    if N <= 0:
        return 0
    total = 6 % mod  # n=1
    if N == 1:
        return total

    k_max = icbrt_floor(N - 1)
    for k in range(1, k_max + 1):
        k3 = k * k * k
        if k3 + 1 > N:
            break
        full = (k + 1) * (k + 1) * (k + 1) - k3
        L = min(N - k3, full)

        k2 = k * k
        cap2 = k * (k + 1)

        lenA = min(L, k2)
        rem = L - lenA
        lenB = min(rem, cap2) if rem > 0 else 0
        rem -= lenB
        lenC = rem if rem > 0 else 0

        c_k2 = c_count(k2)
        c_kk1 = c_count(cap2)

        # Sum of (c(pz)+c(qz)+c(rz)) across the block
        sum_c = c_prefix_sum(lenA)
        if lenB:
            sum_c += lenB * c_k2 + c_prefix_sum(lenB)
        if lenC:
            sum_c += lenC * (c_k2 + c_kk1) + c_prefix_sum(lenC)

        # Sum of the per-n constant term bv (4 / 8 / 12) across the block
        sum_bv = 4 * lenA + 8 * lenB + 12 * lenC

        block = (L * (6 * k2) + sum_bv + 2 * sum_c) % mod
        total += block
        total %= mod

    return total


def G_mod(N: int, mod: int = MOD) -> int:
    """Compute G(N) modulo mod."""
    # sum_{n=1..N} 6n = 3N(N+1)
    sum6 = (3 * (N % mod) * ((N + 1) % mod)) % mod
    S = sum_smin_mod(N, mod)
    return (sum6 - S) % mod


def _self_test() -> None:
    # Test values from the problem statement
    assert g(10) == 30
    assert g(18) == 66
    assert G_mod(18) == 530
    assert G_mod(10**6) == 951640919


def main() -> None:
    _self_test()
    print(G_mod(10**16))


if __name__ == "__main__":
    main()
