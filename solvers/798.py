#!/usr/bin/env python3
"""
Project Euler 798 — Card Stacking Game

Compute C(n, s) modulo 1_000_000_007.

No external libraries are used (only Python standard library).
"""

from array import array

MOD = 1_000_000_007


def _build_factorials(N: int, mod: int = MOD):
    """fact[i] = i! mod, inv_fact[i] = (i!)^{-1} mod for 0<=i<=N."""
    fact = array("I", [1]) * (N + 1)
    # factorials
    for i in range(2, N + 1):
        fact[i] = (fact[i - 1] * i) % mod

    inv_fact = array("I", [1]) * (N + 1)
    inv_fact[N] = pow(int(fact[N]), mod - 2, mod)
    for i in range(N, 0, -1):
        inv_fact[i - 1] = (inv_fact[i] * i) % mod
    return fact, inv_fact


def _nCk(n: int, k: int, fact: array, inv_fact: array, mod: int = MOD) -> int:
    """Binomial coefficient n choose k modulo mod (0 if out of range)."""
    if k < 0 or k > n:
        return 0
    return (int(fact[n]) * int(inv_fact[k]) % mod) * int(inv_fact[n - k]) % mod


def _fwht_xor_inplace(a: array, mod: int = MOD) -> None:
    """
    In-place Walsh–Hadamard transform over XOR, modulo mod.

    a must contain values in [0, mod).
    """
    n = len(a)
    h = 1
    while h < n:
        step = h << 1
        for i in range(0, n, step):
            j = i
            end = i + h
            while j < end:
                x = int(a[j])
                y = int(a[j + h])
                u = x + y
                if u >= mod:
                    u -= mod
                v = x - y
                if x < y:
                    v += mod
                a[j] = u
                a[j + h] = v
                j += 1
        h = step


def _build_single_suit_distribution_padded(n: int, L: int, mod: int = MOD) -> array:
    """
    Build f[0..L-1], where f[g] = number of initial visible sets for a single suit
    whose Grundy value equals g, modulo mod. Values for g>=n are 0.

    This uses the recurrence described in README (diagonal DP for a weighted
    negative-binomial sum), in O(n) time and O(n) memory for factorials.
    """
    a = array("I", [0]) * L
    if n <= 0:
        return a
    if n == 1:
        a[0] = 2  # empty set or {1}; both losing, hence Grundy 0
        return a

    # factorial range needed: max binom argument is <= n
    fact, inv_fact = _build_factorials(n, mod)

    # base cases
    a0 = (pow(2, n - 2, mod) + 2) % mod
    a1 = (pow(2, n - 2, mod) + (n - 2)) % mod
    a[0] = a0
    if n > 1:
        a[1] = a1
    if n > 2:
        a[2] = (pow(2, n - 3, mod) + (n - 3)) % mod

    inv4 = pow(4, mod - 2, mod)

    # Helper: Q(X,k) = X*C(X+k+1,k+1) - (k+1)*C(X+k+1,k+2)
    def Q_of(X: int, k: int) -> int:
        n1 = X + k + 1
        c1 = _nCk(n1, k + 1, fact, inv_fact, mod)
        c2 = _nCk(n1, k + 2, fact, inv_fact, mod)
        return (X * c1 - (k + 1) * c2) % mod

    # We need F(X,k) = sum_{r=0..X} C(r+k,k) * 2^(X-r)
    # along diagonals where X decreases by 2 as k increases by 1.
    #
    # Update:
    #   F(X-2,k) = (F(X,k) - 2*C(X+k-1,k) - C(X+k,k)) / 4
    #   F(X-2,k+1) = 2*F(X-2,k) - C(X+k-1,k+1)
    #
    # All arithmetic is modulo mod (division via inv4).

    # Odd Grundy values: g = 2k + 3, starting at (X0=n-4, k=0)
    X0 = n - 4
    if 3 < n and X0 >= 0:
        k = 0
        X = X0
        F = (pow(2, X + 1, mod) - 1) % mod  # F(X,0) = 2^(X+1)-1
        while True:
            g = 2 * k + 3
            if g >= n or X < 0:
                break
            a[g] = (F + Q_of(X, k)) % mod

            if X < 2:
                break  # next would need X-2 >= 0
            # Compute F(X-2,k)
            c_xk_1 = _nCk(X + k - 1, k, fact, inv_fact, mod)  # C(X+k-1,k)
            c_xk = _nCk(X + k, k, fact, inv_fact, mod)  # C(X+k,k)
            tmp = (F - 2 * c_xk_1 - c_xk) % mod
            tmp = (tmp * inv4) % mod
            # Compute F(X-2,k+1)
            c_next = _nCk(X + k - 1, k + 1, fact, inv_fact, mod)  # C(X+k-1,k+1)
            F = (2 * tmp - c_next) % mod
            k += 1
            X -= 2

    # Even Grundy values: g = 2k + 4, starting at (X0=n-5, k=0)
    X0 = n - 5
    if 4 < n and X0 >= 0:
        k = 0
        X = X0
        F = (pow(2, X + 1, mod) - 1) % mod  # F(X,0)
        while True:
            g = 2 * k + 4
            if g >= n or X < 0:
                break
            a[g] = (F + Q_of(X, k)) % mod

            if X < 2:
                break
            c_xk_1 = _nCk(X + k - 1, k, fact, inv_fact, mod)
            c_xk = _nCk(X + k, k, fact, inv_fact, mod)
            tmp = (F - 2 * c_xk_1 - c_xk) % mod
            tmp = (tmp * inv4) % mod
            c_next = _nCk(X + k - 1, k + 1, fact, inv_fact, mod)
            F = (2 * tmp - c_next) % mod
            k += 1
            X -= 2

    return a


def compute_C(n: int, s: int, mod: int = MOD) -> int:
    """
    Compute C(n,s) modulo mod.

    Uses XOR-convolution via Walsh–Hadamard transform:
      C(n,s) = (1/L) * sum_t (Fhat[t])^s  mod,
    where L is the next power of two >= n, and Fhat is the WHT of the single-suit
    Grundy distribution padded to length L.
    """
    if n < 0 or s < 0:
        raise ValueError("n and s must be non-negative")

    if n == 0:
        # Empty deck: only the empty initial set exists and first player loses.
        return 1

    L = 1 << (n - 1).bit_length()
    f = _build_single_suit_distribution_padded(n, L, mod)

    _fwht_xor_inplace(f, mod)

    # Pointwise exponentiation and sum; reduce mod periodically to save time.
    total = 0
    mask = 8191  # periodic mod to reduce overhead
    for i, v in enumerate(f):
        total += pow(int(v), s, mod)
        if (i & mask) == 0:
            total %= mod
    total %= mod

    inv_L = pow(L, mod - 2, mod)
    return (total * inv_L) % mod


def _self_test() -> None:
    # Test values from the problem statement
    assert compute_C(3, 2) == 26
    assert compute_C(13, 4) == 540318329


def main() -> None:
    _self_test()
    n = 10_000_000
    s = 10_000_000
    print(compute_C(n, s))


if __name__ == "__main__":
    main()
