#!/usr/bin/env python3
# Project Euler 759: A Squared Recurrence Relation
#
# We use the identity f(n) = n * popcount(n), then compute
# S(N) = sum_{i=1..N} (i^2 * popcount(i)^2) mod 1_000_000_007
# via a bit-DP that aggregates moments over ranges.

MOD = 1_000_000_007


def _zero_mat():
    # mat[pc_power][degree] for pc_power in {0,1,2} and degree in {0,1,2}
    return [[0, 0, 0], [0, 0, 0], [0, 0, 0]]


def _add_mat(a, b):
    return [
        [
            (a[0][0] + b[0][0]) % MOD,
            (a[0][1] + b[0][1]) % MOD,
            (a[0][2] + b[0][2]) % MOD,
        ],
        [
            (a[1][0] + b[1][0]) % MOD,
            (a[1][1] + b[1][1]) % MOD,
            (a[1][2] + b[1][2]) % MOD,
        ],
        [
            (a[2][0] + b[2][0]) % MOD,
            (a[2][1] + b[2][1]) % MOD,
            (a[2][2] + b[2][2]) % MOD,
        ],
    ]


def _shift_range(mat, p):
    """
    Given aggregates over y in [0..R]:
        mat[t][d] = sum y^d * popcount(y)^t
    return aggregates over x = p + y:
        out[j][d] = sum (p+y)^d * (1+popcount(y))^j
    for j in {0,1,2}, d in {0,1,2}.
    """
    p %= MOD
    p2 = (p * p) % MOD
    two_p = (2 * p) % MOD

    # For each t, compute sums of (p+y)^d * pc(y)^t for d=0..2
    mats = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for t in (0, 1, 2):
        s0 = mat[t][0] % MOD  # sum pc^t
        s1 = mat[t][1] % MOD  # sum y * pc^t
        s2 = mat[t][2] % MOD  # sum y^2 * pc^t
        mats[t][0] = s0
        mats[t][1] = (p * s0 + s1) % MOD
        mats[t][2] = (p2 * s0 + two_p * s1 + s2) % MOD

    # (1+pc)^j coefficients for pc^t, t=0..2
    # j=0: 1
    # j=1: 1 + pc
    # j=2: 1 + 2pc + pc^2
    coeffs = (
        (1, 0, 0),
        (1, 1, 0),
        (1, 2, 1),
    )

    out = _zero_mat()
    for j in (0, 1, 2):
        c0, c1, c2 = coeffs[j]
        for d in (0, 1, 2):
            out[j][d] = (c0 * mats[0][d] + c1 * mats[1][d] + c2 * mats[2][d]) % MOD
    return out


def _precompute_full(max_bits):
    """
    full[m] is the aggregate matrix over [0 .. 2^m - 1].
    In particular, full[0] is over [0..0].
    """
    full = [_zero_mat() for _ in range(max_bits + 1)]
    full[0][0][0] = 1  # y^0 * pc(y)^0 over y=0: count = 1
    for m in range(1, max_bits + 1):
        p = 1 << (m - 1)
        full[m] = _add_mat(full[m - 1], _shift_range(full[m - 1], p))
    return full


def _calc_upto(n, full):
    """
    Return aggregate matrix over [0 .. n] using:
      [0..n] = [0..2^k-1] U [2^k..n] where 2^k is highest power of two <= n
    """
    if n < 0:
        return _zero_mat()
    if n == 0:
        return full[0]

    k = n.bit_length() - 1
    p = 1 << k
    if n == p - 1:
        return full[k]

    r = n - p
    return _add_mat(full[k], _shift_range(_calc_upto(r, full), p))


def S_mod(n, full):
    # S(n) = sum_{i=1..n} f(i)^2, and f(i) = i*popcount(i)
    # so S(n) = sum_{i=1..n} i^2 * popcount(i)^2.
    return _calc_upto(n, full)[2][2] % MOD


def main():
    # Test values from the problem statement
    targets = [10, 10**2, 10**16]
    max_bits = max(t.bit_length() for t in targets)
    full = _precompute_full(max_bits)

    assert S_mod(10, full) == 1530
    assert S_mod(10**2, full) == 4798445

    print(S_mod(10**16, full))


if __name__ == "__main__":
    main()
