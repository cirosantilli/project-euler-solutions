#!/usr/bin/env python3
"""
Project Euler 702 - Jumping Flea

Compute S(N) for the problem's N (default: 123456789).

The implementation is based on reducing the counting task to inversion counting in
a modular-multiplication permutation, and evaluating those inversion counts using
a fast Euclidean-style recursion.

No third-party libraries are used.
"""

import sys


def _inv_count_mod_mult(x: int, m: int, memo: dict[tuple[int, int], int]) -> int:
    """
    f(x, m): number of inversions in the sequence:
        a*x mod m  for a = 1..m-1
    assuming gcd(x, m) = 1.

    Runs in about O(log m) using a recursion derived from the Euclidean algorithm.
    """
    if m <= 2:
        return 0

    x %= m
    if x == 0:
        # This would violate gcd(x, m)=1 for m>1; keep safe for callers.
        return 0
    if x == 1:
        return 0
    if x == m - 1:
        # Perfect reversal of 1..m-1
        return (m - 1) * (m - 2) // 2

    key = (x, m)
    if key in memo:
        return memo[key]

    t = m // x
    y = m - t * x  # m = t*x + y, with 0 < y < x when gcd(x,m)=1 and x>1

    # t(t+1)x(x-1)/4 computed as integer without fractions:
    #   (t(t+1)/2) * (x(x-1)/2)
    block = (t * (t + 1) // 2) * (x * (x - 1) // 2)

    # Note: f(x, y) and f(x, x-y) are valid because gcd(x,y)=gcd(x,x-y)=1.
    res = (
        block
        + (t + 1) * _inv_count_mod_mult(x, y, memo)
        - t * _inv_count_mod_mult(x, x - y, memo)
    )

    memo[key] = res
    return res


def _g(x: int, m: int, memo: dict[tuple[int, int], int]) -> int:
    """
    g(x, m) = (m-1)(m-2) - f(x, m)
    where f is inversion count for the modular multiplication permutation.
    """
    if m <= 2:
        return 0
    return (m - 1) * (m - 2) - _inv_count_mod_mult(x, m, memo)


def S(N: int) -> int:
    """
    Return S(N) as defined in the statement.

    The derivation yields:
        D = bit_length(N)  (i.e. floor(log2 N) + 1 for N>0)
        S(N) = N(3N+1)/2 * (D+1)
               - sum_{d=2..D} g(N, 2^d)
               + 2 * g(N, 2^D - N)

    This formulation assumes N is odd (so gcd(N, 2^d)=1).
    """
    if N <= 0:
        raise ValueError("N must be positive.")
    if N % 2 == 0:
        raise ValueError("This method assumes N is odd (coprime to powers of two).")

    memo: dict[tuple[int, int], int] = {}
    D = N.bit_length()

    base = (N * (3 * N + 1) // 2) * (D + 1)

    total = base
    for d in range(2, D + 1):
        total -= _g(N, 1 << d, memo)
    total += 2 * _g(N, (1 << D) - N, memo)

    return total


def _self_test() -> None:
    # Test values given in the problem statement
    assert S(3) == 42
    assert S(5) == 126
    assert S(123) == 167178
    assert S(12345) == 3185041956


def main(argv: list[str]) -> None:
    _self_test()

    if len(argv) >= 2:
        N = int(argv[1])
    else:
        N = 123456789

    print(S(N))


if __name__ == "__main__":
    main(sys.argv)
