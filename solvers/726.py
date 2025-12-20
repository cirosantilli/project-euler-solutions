#!/usr/bin/env python3
"""
Project Euler 726: Falling Bottles

Compute S(10^4) modulo 1_000_000_033.

No external libraries are used.
"""

MOD = 1_000_000_033
TARGET_N = 10_000


def _compute_f_values_upto(n: int, mod: int = MOD):
    """
    Returns list f where f[k] = f(k) mod mod for 0<=k<=n.
    Uses the closed-form factorization described in README.md, implemented via an O(n^2)
    (but ~n(n+1)/2 multiplications) incremental product.
    """
    if n < 1:
        return [0]

    inv_odd = [0] * (n + 1)
    # inv_odd[k] = (2k-1)^(-1) mod mod
    for k in range(1, n + 1):
        inv_odd[k] = pow(2 * k - 1, mod - 2, mod)

    f = [0] * (n + 1)
    f[1] = 1 % mod

    curN = 1  # N_1 = 1*2/2
    pow2 = 2 % mod  # 2^1
    mers_prefix = 1 % mod  # ∏_{i=1..1} (2^i-1) = 1
    odd_inv_prefix = 1 % mod  # ∏_{i=1..1} (2i-1)^(-1) = 1
    cur_f = 1 % mod

    for layer in range(2, n + 1):
        # Multiply by (N_{layer-1}+1) ... N_layer, where N_layer = layer(layer+1)/2.
        start = curN + 1
        end = curN + layer
        for x in range(start, end + 1):
            cur_f = (cur_f * x) % mod
        curN = end

        # Update prefixes:
        pow2 = (pow2 * 2) % mod  # now 2^layer
        mers_prefix = (mers_prefix * (pow2 - 1)) % mod  # multiply by (2^layer - 1)
        odd_inv_prefix = (
            odd_inv_prefix * inv_odd[layer]
        ) % mod  # multiply by (2*layer-1)^(-1)

        # Apply the ratio factor:
        cur_f = (cur_f * mers_prefix) % mod
        cur_f = (cur_f * odd_inv_prefix) % mod

        f[layer] = cur_f

    return f


def solve(n: int = TARGET_N, mod: int = MOD) -> int:
    """
    Computes S(n) = sum_{k=1..n} f(k) (mod mod).
    """
    if n < 1:
        return 0

    # Precompute inverses of odd numbers up to 2n-1 (only n of them are needed).
    inv_odd = [0] * (n + 1)
    for k in range(1, n + 1):
        inv_odd[k] = pow(2 * k - 1, mod - 2, mod)

    cur_f = 1 % mod
    total = cur_f
    curN = 1  # N_1
    pow2 = 2 % mod
    mers_prefix = 1 % mod
    odd_inv_prefix = 1 % mod

    for layer in range(2, n + 1):
        start = curN + 1
        end = curN + layer
        for x in range(start, end + 1):
            cur_f = (cur_f * x) % mod
        curN = end

        pow2 = (pow2 * 2) % mod
        mers_prefix = (mers_prefix * (pow2 - 1)) % mod
        odd_inv_prefix = (odd_inv_prefix * inv_odd[layer]) % mod

        cur_f = (cur_f * mers_prefix) % mod
        cur_f = (cur_f * odd_inv_prefix) % mod

        total += cur_f
        total %= mod

    return total


def _self_test():
    # Test values given in the problem statement:
    f = _compute_f_values_upto(3, MOD)
    assert f[1] == 1
    assert f[2] == 6
    assert f[3] == 1008


if __name__ == "__main__":
    _self_test()
    print(solve())
