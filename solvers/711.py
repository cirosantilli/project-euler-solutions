#!/usr/bin/env python3
"""
Project Euler 711 - Binary Blackboard

Compute S(N) modulo 1_000_000_007, where S(N) is the sum of all n <= 2^N for which
Eric can guarantee winning (optimal play).

No external libraries are used.
"""

import sys


MOD = 1_000_000_007
INV3 = pow(3, MOD - 2, MOD)
INV7 = pow(7, MOD - 2, MOD)


def _sum_pows_8(k: int) -> int:
    """Return sum_{i=0..k-1} 8^i (mod MOD)."""
    if k <= 0:
        return 0
    return (pow(8, k, MOD) - 1) * INV7 % MOD


def _sum_pows_4_1_to_m(m: int) -> int:
    """Return sum_{i=1..m} 4^i (mod MOD)."""
    if m <= 0:
        return 0
    return (pow(4, m + 1, MOD) - 4) * INV3 % MOD


def _sum_T_upto(K: int) -> int:
    """
    Let B_k be the offset-set described in the solution, and let T_k = sum_{b in B_k} b (mod MOD).
    This returns sum_{k=0..K} T_k (mod MOD). Note T_0 = 0.
    """
    if K <= 0:
        return 0

    mod = MOD
    T = 0
    s = 0
    pow2 = 1  # 2^0
    pow4 = 1  # 4^0

    # We iterate k = 0..K-1 and build T_{k+1} from T_k.
    for _ in range(K):
        add = (pow2 + 2) * pow4 % mod
        T = (T + T + add) % mod  # 2*T + add
        s += T
        if s >= mod:
            s -= mod
        pow2 = (pow2 + pow2) % mod
        pow4 = (pow4 * 4) % mod

    return s


def S(N: int) -> int:
    """
    Compute S(N) modulo MOD.

    The winning n have a base-4 structure that leads to:
      S(2m)   = A(m-1) + 4^m + B(m)
      S(2m+1) = A(m)   + B(m)
    where:
      A(t) = sum_{k=0..t} (2^k * 4^k + T_k) = sum_{k=0..t} 8^k + sum_{k=0..t} T_k
      B(m) = sum_{k=1..m} (4^k - 1)
    """
    if N < 0:
        raise ValueError("N must be non-negative")

    if N % 2 == 0:
        m = N // 2

        # A(m-1) = sum_{k=0..m-1} 8^k + sum_{k=0..m-1} T_k
        A = (_sum_pows_8(m) + _sum_T_upto(m - 1)) % MOD

        pow4m = pow(4, m, MOD)
        B = (_sum_pows_4_1_to_m(m) - m) % MOD

        return (A + pow4m + B) % MOD

    else:
        m = (N - 1) // 2

        # A(m) = sum_{k=0..m} 8^k + sum_{k=0..m} T_k
        A = (_sum_pows_8(m + 1) + _sum_T_upto(m)) % MOD

        B = (_sum_pows_4_1_to_m(m) - m) % MOD

        return (A + B) % MOD


def main() -> None:
    # Tests given in the problem statement:
    assert S(4) == 46
    assert S(12) == 54532
    assert S(1234) == 690421393

    N = 12_345_678
    if len(sys.argv) >= 2:
        N = int(sys.argv[1])

    print(S(N))


if __name__ == "__main__":
    main()
