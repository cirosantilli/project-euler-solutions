#!/usr/bin/env python3
"""Project Euler 877: XOR-Equation A

We work with the XOR-product (carryless multiplication over GF(2)).

The core observation is that all (a,b) solutions to

    (a ⊗ a) ⊕ (2 ⊗ a ⊗ b) ⊕ (b ⊗ b) = 5

with 0 <= a <= b form a single increasing sequence of consecutive pairs:

    (s0, s1), (s1, s2), (s2, s3), ...

where s0 = 0, s1 = 3 and

    s_{n+2} = (s_{n+1} << 1) XOR s_n.

Therefore X(N) is just XOR of all s_k (k>=1) with s_k <= N.

The program prints X(10^18) by default, or X(N) if N is supplied as
an integer command line argument.
"""

import sys


def xor_product(x: int, y: int) -> int:
    """Carryless multiplication (GF(2)[z]) of nonnegative integers."""
    if x < 0 or y < 0:
        raise ValueError("xor_product is defined here for nonnegative integers only")
    res = 0
    while y:
        if y & 1:
            res ^= x
        y >>= 1
        x <<= 1
    return res


def equation_value(a: int, b: int) -> int:
    """Compute (a⊗a) ⊕ (2⊗a⊗b) ⊕ (b⊗b)."""
    # 2 ⊗ t is just a left shift by 1 in carryless multiplication.
    return xor_product(a, a) ^ (xor_product(a, b) << 1) ^ xor_product(b, b)


def X(N: int) -> int:
    """X(N): XOR of b over all solutions with 0 <= a <= b <= N."""
    if N < 0:
        raise ValueError("N must be nonnegative")

    # Generate the unique increasing solution ladder:
    # (s0,s1)=(0,3), (s1,s2), ... with s_{n+2} = (s_{n+1}<<1) XOR s_n.
    s_prev, s_cur = 0, 3
    acc = 0
    while s_cur <= N:
        acc ^= s_cur
        s_prev, s_cur = s_cur, (s_cur << 1) ^ s_prev
    return acc


def _self_test() -> None:
    # Test values explicitly given in the problem statement.
    assert xor_product(7, 3) == 9
    assert equation_value(3, 6) == 5
    assert X(10) == 5


def main() -> None:
    _self_test()

    if len(sys.argv) >= 2:
        N = int(sys.argv[1])
    else:
        N = 10**18

    print(X(N))


if __name__ == "__main__":
    main()
