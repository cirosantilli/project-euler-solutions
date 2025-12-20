#!/usr/bin/env python3
"""
Project Euler 741 - Binary Grid Colouring

Counts n×n 0/1 grids with exactly two 1s in every row and column, up to D4 symmetries.
No external libraries.
"""

MOD = 1_000_000_007
INV2 = (MOD + 1) // 2
INV8 = pow(8, MOD - 2, MOD)


def _f_diag_fact(n: int) -> tuple[int, int, int]:
    """
    Returns (f(n) mod MOD, diag_fix(n) mod MOD, n! mod MOD).

    f(n): total count of n×n grids with row/col sums = 2.
    diag_fix(n): number fixed by main-diagonal reflection (transpose).
    """
    mod = MOD
    inv2 = INV2

    # factorial n!
    fact = 1

    # Sequence h_n = n! * a_n where a(x) = exp(-x/2) / sqrt(1-x) and
    # f(n) = n! * h_n  (equivalently f(n) = (n!)^2 * [x^n] a(x)).
    # Recurrence (for n>=1): h_{n+1} = n*h_n + (n/2)*h_{n-1}
    h_im2 = 1  # h_0
    h_im1 = 0  # h_1

    # Diagonal-fix sequence A_n (integer) with:
    # A_0=1, A_1=0, A_2=1, A_3=4 and for k>=3:
    # A_{k+1} = 2k*A_k - k(k-2)*A_{k-1} - (k(k-1)(k-2)/2)*A_{k-3}
    if n == 0:
        return 1, 1, 1
    if n == 1:
        # fact will become 1 after loop
        pass

    if n <= 3:
        # We'll compute fact and h in the main loop and return the small diagonal values after.
        d0 = d1 = d2 = d3 = 0
    else:
        d0, d1, d2, d3 = 1, 0, 1, 4  # A_0..A_3

    for i in range(1, n + 1):
        fact = (fact * i) % mod

        if i >= 2:
            k = i - 1
            h_i = (k * h_im1 + (k * inv2 % mod) * h_im2) % mod
            h_im2, h_im1 = h_im1, h_i

        if n >= 4 and i >= 4:
            k = i - 1
            # window: d0=A_{i-4}, d1=A_{i-3}, d2=A_{i-2}, d3=A_{i-1}
            term1 = (2 * k * d3) % mod
            term2 = (k * (k - 2)) % mod
            term2 = (term2 * d2) % mod
            term3 = (k * (k - 1)) % mod
            term3 = (term3 * (k - 2)) % mod
            term3 = (term3 * d0) % mod
            new = (term1 - term2 - (term3 * inv2 % mod)) % mod
            d0, d1, d2, d3 = d1, d2, d3, new

    f_n = (fact * h_im1) % mod  # f(n) = n! * h_n

    if n == 0:
        diag = 1
    elif n == 1:
        diag = 0
    elif n == 2:
        diag = 1
    elif n == 3:
        diag = 4
    else:
        diag = d3

    return f_n, diag, fact


def _fix_axis_reflection(n: int, fact_n: int) -> int:
    """Fixed count for vertical (or horizontal) reflection."""
    if n & 1:
        return 0
    # n! / 2^{n/2}
    return fact_n * pow(INV2, n // 2, MOD) % MOD


def _fix_rotation_90(n: int) -> int:
    """Fixed count for 90° rotation (same as 270°)."""
    if n & 1:
        return 0
    m = n // 2

    # b_m counts 2-regular multigraphs on m labeled vertices with:
    # - loops allowed,
    # - between distinct vertices, an edge has 2 "types",
    # - a double-edge component uses both types (unique).
    # EGF simplifies to (1-2x)^(-1/2) * exp(-x^2/2),
    # leading to an integer recurrence for b_n:
    # b_0=1, b_1=1, b_2=2 and for n>=2:
    # b_{n+1} = (2n+1)b_n - n b_{n-1} + 2n(n-1)b_{n-2}.
    if m == 0:
        return 1
    if m == 1:
        return 1
    if m == 2:
        return 2

    mod = MOD
    b0, b1, b2 = 1, 1, 2  # b_{n-2}, b_{n-1}, b_n with n=2
    for i in range(2, m):
        val = ((2 * i + 1) * b2 - i * b1 + (2 * i * (i - 1) % mod) * b0) % mod
        b0, b1, b2 = b1, b2, val
    return b2


def _fix_rotation_180(n: int) -> int:
    """Fixed count for 180° rotation."""
    mod = MOD
    if n == 0:
        return 1

    if (n & 1) == 0:
        # even n = 2m
        m = n // 2
        if m == 0:
            return 1
        if m == 1:
            return 1

        # j_n = n! * e_n where E(x)=exp(-x)/sqrt(1-4x), with integer recurrence:
        # j_0=1, j_1=1, j_{n+1} = (4n+1)j_n + 4n j_{n-1}
        j_prev, j_curr = 1, 1
        fact = 1  # will end as (m-1)!
        for i in range(1, m):
            fact = (fact * i) % mod
            j_next = ((4 * i + 1) * j_curr + (4 * i % mod) * j_prev) % mod
            j_prev, j_curr = j_curr, j_next

        fact_m = (fact * m) % mod
        return fact_m * j_curr % mod

    # odd n = 2m+1
    m = (n - 1) // 2
    if m == 0:
        return 0

    # We need t_m = m! * s_m where s is defined by D(x)=2xE(x)/(1-4x).
    # With j_n above, t has integer recurrence:
    # t_0=0,  t_n = 4n t_{n-1} + 2n j_{n-1}.
    fact = 1
    t = 0
    j_prev, j_curr = 1, 1  # j_0, j_1
    for i in range(1, m + 1):
        fact = (fact * i) % mod
        t = ((4 * i % mod) * t + (2 * i % mod) * j_prev) % mod
        if i < m:
            j_next = ((4 * i + 1) * j_curr + (4 * i % mod) * j_prev) % mod
            j_prev, j_curr = j_curr, j_next

    return fact * t % mod


def g(n: int) -> int:
    """
    Number of valid n×n grids, up to rotations and reflections (D4).
    """
    f_n, diag, fact_n = _f_diag_fact(n)
    axis = _fix_axis_reflection(n, fact_n)
    r180 = _fix_rotation_180(n)
    r90 = _fix_rotation_90(n)

    total = (f_n + r180 + 2 * r90 + 2 * axis + 2 * diag) % MOD
    return total * INV8 % MOD


def main() -> None:
    # Problem statement test values
    assert _f_diag_fact(4)[0] == 90
    assert _f_diag_fact(7)[0] == 3_110_940
    assert _f_diag_fact(8)[0] == 187_530_840

    assert g(4) == 20
    assert g(7) == 390_816
    assert g(8) == 23_462_347
    assert (g(7) + g(8)) == 23_853_163

    n1 = 7**7
    n2 = 8**8
    ans = (g(n1) + g(n2)) % MOD
    print(ans)


if __name__ == "__main__":
    main()
