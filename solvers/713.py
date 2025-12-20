#!/usr/bin/env python3
"""
Project Euler 713: Turán's Water Heating System

Compute L(10^7) efficiently using:
- Turán's theorem interpretation for T(N, m)
- Divisor-summatory grouping to sum T over m in O(sqrt(N))
"""


def t_min_tries(N: int, m: int) -> int:
    """
    T(N, m): smallest number of fuse-pair tests required to guarantee finding
    a working pair, given N fuses of which exactly m are working.

    Optimal strategy corresponds to testing all pairs within each part of a
    balanced (m-1)-part partition of the N fuses (i.e., complement of a Turán graph).
    """
    if not (2 <= m <= N):
        raise ValueError("Require 2 <= m <= N")
    k = m - 1  # number of parts
    q, r = divmod(N, k)  # r parts of size (q+1), k-r parts of size q

    # Sum of C(size, 2) over parts:
    # r * C(q+1, 2) + (k-r) * C(q, 2)
    # = k*C(q,2) + r*q
    return k * q * (q - 1) // 2 + r * q


def L(N: int) -> int:
    """
    L(N) = sum_{m=2..N} T(N, m)

    Uses grouping over constant floor(N/k) where k = m-1, yielding O(sqrt(N)).
    """
    if N < 2:
        return 0

    total = 0
    k = 1
    k_end = N - 1

    while k <= k_end:
        q = N // k
        k_max = N // q
        if k_max > k_end:
            k_max = k_end

        cnt = k_max - k + 1
        sum_k = (k + k_max) * cnt // 2  # arithmetic series

        # T(k) = N*q - k*q*(q+1)/2  where q = floor(N/k)
        total += cnt * N * q - (q * (q + 1) // 2) * sum_k

        k = k_max + 1

    return total


def _self_test() -> None:
    # Test values from the problem statement
    assert t_min_tries(3, 2) == 3
    assert t_min_tries(8, 4) == 7
    assert L(10**3) == 3281346


def main() -> None:
    _self_test()
    print(L(10**7))


if __name__ == "__main__":
    main()
