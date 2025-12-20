#!/usr/bin/env python3
"""
Project Euler 716 - Grid Graphs

We have an H×W lattice of nodes. Each row's horizontal edges share one direction
(left or right), and each column's vertical edges share one direction (up or down).
For each of the 2^(H+W) choices, let S(G) be the number of strongly connected
components. This program computes:

    C(H, W) = sum_G S(G)   (mod 1_000_000_007)

No external libraries are used.
"""

import sys

MOD = 1_000_000_007


def C_mod(h: int, w: int, mod: int = MOD) -> int:
    """
    Compute C(h, w) modulo `mod` using the closed form.

    The expression is symmetric in (h, w):
      C(H,W) =
        9*2^(H+W)
        + 2*H*W*(2^H + 2^W + 1)
        - 8*(W*2^H + H*2^W)
        - 10*(2^H + 2^W)
        + 10*(H + W + 1)
    """
    if h <= 0 or w <= 0:
        raise ValueError("H and W must be positive integers")

    pow2h = pow(2, h, mod)
    pow2w = pow(2, w, mod)
    pow2hw = (pow2h * pow2w) % mod

    term1 = (9 * pow2hw) % mod
    term2 = (2 * (h % mod) * (w % mod)) % mod
    term2 = (term2 * ((pow2h + pow2w + 1) % mod)) % mod

    term3 = (-8 * ((w % mod) * pow2h + (h % mod) * pow2w)) % mod
    term4 = (-10 * (pow2h + pow2w)) % mod
    term5 = (10 * ((h + w + 1) % mod)) % mod

    return (term1 + term2 + term3 + term4 + term5) % mod


def _self_test() -> None:
    # Test values from the problem statement
    assert C_mod(3, 3) == 408
    assert C_mod(3, 6) == 4696
    assert C_mod(10, 20) == 988971143


def main() -> None:
    _self_test()

    if len(sys.argv) == 3:
        h = int(sys.argv[1])
        w = int(sys.argv[2])
    else:
        h, w = 10_000, 20_000

    print(C_mod(h, w))


if __name__ == "__main__":
    main()
