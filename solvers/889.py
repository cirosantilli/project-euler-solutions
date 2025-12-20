#!/usr/bin/env python3
"""Project Euler 889: Rational Blancmange

Compute
    F(k, t, r) = (2^(2k) - 1) * T( ((2^t + 1)^r) / (2^k + 1) )
where
    T(x) = sum_{n>=0} s(2^n x)/2^n
and s(y) is the distance from y to the nearest integer.

The required output is:
    F(10^18 + 31, 10^14 + 31, 62) mod 1_000_062_031.

No external libraries are used.
"""

from __future__ import annotations

from bisect import bisect_right
from math import comb

MOD = 1_000_062_031


def _iter_set_bits(n: int):
    """Yield positions of set bits in n (least-significant bit is position 0)."""
    while n:
        lsb = n & -n
        yield lsb.bit_length() - 1
        n -= lsb


def _bit_positions_of_N_sparse(t: int, r: int) -> list[int]:
    """Bit-positions of N = (2^t + 1)^r, without building N.

    Uses the binomial expansion:
        (2^t + 1)^r = sum_{u=0..r} C(r,u) * 2^{t*u}.

    This is safe when every binomial coefficient fits inside the t-bit window
    (so blocks don't overlap/carry across windows).
    """
    max_coeff_bitlen = comb(r, r // 2).bit_length()
    if t <= max_coeff_bitlen:
        raise ValueError(
            "t too small for sparse expansion without carries; "
            "use the brute solver for small k instead"
        )

    positions: list[int] = []
    for u in range(r + 1):
        c = comb(r, u)
        base = t * u
        for b in _iter_set_bits(c):
            positions.append(base + b)

    positions.sort()
    # sanity: blocks are disjoint under the condition above
    for i in range(1, len(positions)):
        if positions[i] == positions[i - 1]:
            raise AssertionError("duplicate bit position produced")
    return positions


def _brute_F_mod(k: int, t: int, r: int, mod: int = MOD) -> int:
    """Brute computation of F(k,t,r) mod mod.

    This uses the identity (valid for denominator 2^k+1):
        F = sum_{j=0..k-1} d_j * 2^{k-j}
    where d_j = min(m_j, (2^k+1) - m_j) and m_j = (2^j * N) mod (2^k+1).

    Intended only for small k (the modulus 2^k+1 is built explicitly).
    """
    Q = (1 << k) + 1
    N_mod = pow(((1 << t) + 1), r, Q)

    m = N_mod
    inv2 = pow(2, -1, mod)  # mod is odd, so inverse exists
    weight = pow(2, k, mod)  # 2^{k-j} with j=0 initially

    ans = 0
    for _j in range(k):
        d = m if m <= Q - m else Q - m
        ans = (ans + (d % mod) * weight) % mod
        weight = (weight * inv2) % mod
        m = (m * 2) % Q

    return ans


def _fast_F_mod(k: int, t: int, r: int, mod: int = MOD) -> int:
    """Fast computation for the large instance in the problem.

    Requires that k is larger than the highest set bit of N = (2^t+1)^r.
    """
    positions = _bit_positions_of_N_sparse(t, r)
    max_pos = positions[-1]
    if k <= max_pos + 1:
        raise ValueError("fast solver requires k > bit_length(N)")

    pow2_k = pow(2, k, mod)

    # precompute 2^p (mod mod) for each set bit position p
    vals_low = [pow(2, p, mod) for p in positions]  # 2^p
    vals_high = [(pow2_k * v) % mod for v in vals_low]  # 2^{k+p} = 2^k * 2^p

    n = len(positions)
    prefix_low = [0] * (n + 1)
    prefix_high = [0] * (n + 1)
    for i in range(n):
        prefix_low[i + 1] = (prefix_low[i] + vals_low[i]) % mod
        prefix_high[i + 1] = (prefix_high[i] + vals_high[i]) % mod

    total_low = prefix_low[n]

    # Base sum assuming d_j = (B_j - C_j) for all j
    # Contribution per bit p: (k-p)*2^{k+p} - p*2^p
    ans = 0
    for p, v_low, v_high in zip(positions, vals_low, vals_high):
        term = ((k - p) % mod) * v_high
        term -= (p % mod) * v_low
        ans = (ans + term) % mod

    # Corrections: for j = k-p0-1 (p0>0), the leading bit forces d_j = (2^k+1) - (B_j-C_j)
    # We can compute the scaled quantity S(p0) = (B_j-C_j) * 2^{k-j} using prefix sums.
    # Then delta = ((2^k+1)*2^{k-j}) - 2*S(p0).
    for i in range(1, n):
        p0 = positions[i]
        # bits <= p0 are in B; bits > p0 are in C
        sum_high_le = prefix_high[i + 1]
        sum_low_gt = (total_low - prefix_low[i + 1]) % mod
        S = (sum_high_le - sum_low_gt) % mod

        pow2_p0_plus1 = (vals_low[i] * 2) % mod  # 2^{p0+1}
        q_times = (pow2_k * pow2_p0_plus1 + pow2_p0_plus1) % mod  # (2^k+1)*2^{p0+1}

        delta = (q_times - 2 * S) % mod
        ans = (ans + delta) % mod

    return ans % mod


def F_mod(k: int, t: int, r: int, mod: int = MOD) -> int:
    """Compute F(k,t,r) modulo mod."""
    # Use brute mode for small k to stay fully general and validate examples.
    if k <= 1500 and t <= 2000:
        return _brute_F_mod(k, t, r, mod)
    return _fast_F_mod(k, t, r, mod)


def main() -> None:
    # Test values from the problem statement
    assert F_mod(3, 1, 1, MOD) == 42
    assert F_mod(13, 3, 3, MOD) == 23_093_880
    assert F_mod(103, 13, 6, MOD) == 878_922_518

    k = 10**18 + 31
    t = 10**14 + 31
    r = 62

    print(F_mod(k, t, r, MOD))


if __name__ == "__main__":
    main()
