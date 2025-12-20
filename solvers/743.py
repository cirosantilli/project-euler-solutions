#!/usr/bin/env python3
"""
Project Euler 743: Window into a Matrix

We count 2×n binary matrices such that every 2×k contiguous window has total sum k.
Answer is required modulo 1_000_000_007.

This solution is specialized for the (large) Euler input, where n is a multiple of k,
and includes asserts for the sample values from the statement.
"""

import sys

MOD = 1_000_000_007

# Tuneable: bigger blocks -> fewer modular exponentiations, more memory.
# 200_000 keeps memory modest while still reducing inversions dramatically.
BLOCK = 200_000


def _inv_range_consecutive(start: int, end: int, mod: int) -> list:
    """
    Return [inv(start), inv(start+1), ..., inv(end)] modulo mod.
    Uses batch inversion in O(L) multiplications and one modular exponentiation,
    where L = end-start+1.

    Precondition: 1 <= start <= end < mod (true for this problem since k < mod).
    """
    L = end - start + 1
    pref = [0] * L

    prod = 1
    x = start
    i = 0
    while i < L:
        prod = (prod * x) % mod
        pref[i] = prod
        x += 1
        i += 1

    inv_prod = pow(prod, mod - 2, mod)

    suffix = inv_prod
    x = end
    i = L - 1
    while i >= 0:
        prev = pref[i - 1] if i else 1
        inv_x = (suffix * prev) % mod
        pref[i] = inv_x
        suffix = (suffix * x) % mod
        x -= 1
        i -= 1

    return pref  # now holds inverses


def A_mod(k: int, n: int, mod: int = MOD) -> int:
    """
    Compute A(k, n) modulo mod.

    For n < k there are no 2×k windows, so the condition is vacuously true: 4^n.
    For n >= k, this implementation assumes n is a multiple of k (as in the Euler input).
    """
    if n < k:
        return pow(4, n, mod)

    if n % k != 0:
        raise NotImplementedError(
            "This implementation handles the Euler input efficiently (n multiple of k)."
        )

    m = n // k  # number of full k-column periods

    # Factor out 2^(m*k), and compute:
    # S = sum_{t=0..floor(k/2)} k!/(t! t! (k-2t)!) * r^t,  where r = 2^(-2m) mod mod
    base = pow(2, m * k, mod)
    r = pow(pow(2, 2 * m, mod), mod - 2, mod)  # 2^{-2m} mod mod

    tmax = k // 2

    term = 1
    acc = 1  # includes t=0

    a = k  # a = k - 2t, updated by -2 each step

    # Iterate t=0..tmax-1, generating term_{t+1} from term_t.
    # term_{t+1} = term_t * (a)*(a-1) * r / (t+1)^2
    # We avoid per-step inversions by generating inverses in blocks.
    start = 1
    while start <= tmax:
        end = start + BLOCK - 1
        if end > tmax:
            end = tmax

        invs = _inv_range_consecutive(start, end, mod)

        # Process this block in forward order.
        for inv in invs:
            invsq_r = (inv * inv) % mod
            invsq_r = (invsq_r * r) % mod

            term = (term * a * (a - 1) * invsq_r) % mod
            acc += term
            a -= 2

        start = end + 1

    return (base * (acc % mod)) % mod


def _self_test() -> None:
    # Test values from the problem statement:
    assert A_mod(3, 9) == 560
    assert A_mod(4, 20) == 1_060_870


def main() -> None:
    _self_test()

    if len(sys.argv) == 3:
        k = int(sys.argv[1])
        n = int(sys.argv[2])
    else:
        k = 10**8
        n = 10**16

    print(A_mod(k, n))


if __name__ == "__main__":
    main()
