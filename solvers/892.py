#!/usr/bin/env python3
"""Project Euler 892: Zebra Circles

Compute:
    S = sum_{n=1}^{10^7} D(n) (mod 1234567891)

No external libraries are used.
"""

from array import array

MOD = 1234567891
N_MAX = 10_000_000


def _modinv(a: int, mod: int = MOD) -> int:
    """Modular inverse (mod is prime)."""
    return pow(a, mod - 2, mod)


def _central_binom_mod(r: int, mod: int = MOD) -> int:
    """Return C(2r, r) modulo mod, using O(r) multiplicative formula."""
    # C(2r,r) = prod_{i=1..r} (r+i)/i
    res = 1
    for i in range(1, r + 1):
        res = (res * (r + i)) % mod
        res = (res * _modinv(i, mod)) % mod
    return res


def D_mod(n: int, mod: int = MOD) -> int:
    """Compute D(n) modulo mod using closed forms."""
    if n <= 0:
        return 0
    inv2 = (mod + 1) // 2
    if n & 1:
        # n = 2r+1
        r = n // 2
        if r == 0:
            return 0
        b = _central_binom_mod(r, mod)  # C(2r,r)
        bb = (b * b) % mod
        return (bb * (2 * r % mod) % mod) * _modinv(r + 1, mod) % mod
    else:
        # n = 2r
        r = n // 2
        b = _central_binom_mod(r, mod)
        bb = (b * b) % mod
        return (bb * inv2) % mod


def solve(n_max: int = N_MAX, mod: int = MOD) -> int:
    """Compute sum_{n=1..n_max} D(n) modulo mod."""
    # We need r up to floor(n_max/2).
    r_max = n_max // 2

    # Precompute inverses inv[i] for i=1..r_max+1 in O(r_max) time.
    inv = array("I", [0]) * (r_max + 2)
    inv[1] = 1
    for i in range(2, r_max + 2):
        inv[i] = (mod - (mod // i) * inv[mod % i] % mod) % mod

    inv2 = (mod + 1) // 2

    b = 1  # b = C(2r, r) for current r, starting at r=0.
    ans = 0

    # n_max is even for this problem, but keep the bounds general.
    for r in range(0, r_max + 1):
        bb = (b * b) % mod

        # Even n = 2r, for r >= 1
        if r:
            ans += bb * inv2
            ans %= mod

        # Odd n = 2r + 1, exists when 2r+1 <= n_max.
        if 0 < r and (2 * r + 1) <= n_max:
            # D(2r+1) = bb * (2r)/(r+1)
            ans += (bb * (2 * r) % mod) * int(inv[r + 1])
            ans %= mod

        # Update b -> C(2(r+1), r+1) using:
        #   C(2(r+1), r+1) = C(2r, r) * (2r+1)(2r+2)/(r+1)^2
        if r < r_max:
            inv_rp1 = int(inv[r + 1])
            b = (b * (2 * r + 1)) % mod
            b = (b * (2 * r + 2)) % mod
            b = (b * inv_rp1) % mod
            b = (b * inv_rp1) % mod

    return ans % mod


def main() -> None:
    # Tests from the problem statement
    assert D_mod(3) == 4
    assert D_mod(100) == 1172122931 % MOD

    print(solve())


if __name__ == "__main__":
    main()
