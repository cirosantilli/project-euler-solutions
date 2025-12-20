#!/usr/bin/env python3
"""
Project Euler 899 - DistribuNim I

Two piles (a, b), both positive.
A move: let s = min(a, b). Remove exactly s stones total from the two piles,
but you may not remove an entire pile. The player with no legal move loses.

This file computes L(n): the number of ordered pairs (a, b) with 1<=a,b<=n
that are losing positions for the first player under optimal play.
"""


def L(n: int) -> int:
    """
    Count losing ordered pairs (a,b) with 1<=a,b<=n.

    Key characterization (proved in README):
    For a<=b, position (a,b) is losing iff b mod 2^k == 2^k-1, where k=bit_length(a).
    """
    if n <= 0:
        return 0

    # Count losing positions with a<=b, then convert to ordered count.
    # U = #{(a,b): 1<=a<=b<=n, (a,b) losing}
    U = 0
    k = 1
    while (1 << (k - 1)) <= n:
        lo = 1 << (k - 1)
        hi = min(n, (1 << k) - 1)
        if hi >= lo:
            A_k = hi - lo + 1  # how many a have bit_length = k

            M = 1 << k
            # C_k = how many b in [1..n] satisfy b ≡ M-1 (mod M)
            if n >= M - 1:
                C_k = (n - (M - 1)) // M + 1
            else:
                C_k = 0

            U += A_k * C_k
        k += 1

    # Diagonal losing positions are exactly a=b=2^k-1.
    D = (n + 1).bit_length() - 1

    # Convert: ordered count = 2*(U - D) + D = 2U - D
    return 2 * U - D


def _self_test() -> None:
    # Test values from the problem statement:
    assert L(7) == 21
    assert L(7 * 7) == 221

    # Minimal sanity:
    assert L(1) == 1  # only (1,1)


def main() -> None:
    _self_test()
    n = pow(7, 17)
    print(L(n))


if __name__ == "__main__":
    main()
