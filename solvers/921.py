#!/usr/bin/env python3
"""Project Euler 921: Golden Recurrence

Computes S(1618034) modulo 398874989.

No external libraries are used.
"""

MOD = 398_874_989
TARGET_M = 1_618_034


def _legendre_symbol(a: int, p: int) -> int:
    """Return the Legendre symbol (a|p) for odd prime p."""
    return pow(a, (p - 1) // 2, p)


def sqrt_mod_prime(n: int, p: int) -> int:
    """Tonelli-Shanks: find x such that x^2 ≡ n (mod p), where p is an odd prime.

    Assumes a solution exists.
    """
    n %= p
    if n == 0:
        return 0
    if p == 2:
        return n
    if _legendre_symbol(n, p) != 1:
        raise ValueError("n is not a quadratic residue modulo p")

    # Simple case
    if p % 4 == 3:
        return pow(n, (p + 1) // 4, p)

    # Factor p-1 = q * 2^s with q odd
    q = p - 1
    s = 0
    while q % 2 == 0:
        q //= 2
        s += 1

    # Find a quadratic non-residue z
    z = 2
    while _legendre_symbol(z, p) != p - 1:
        z += 1

    c = pow(z, q, p)
    x = pow(n, (q + 1) // 2, p)
    t = pow(n, q, p)
    m = s

    while t != 1:
        # Find the smallest i, 0 < i < m, such that t^(2^i) == 1
        i = 1
        t2 = (t * t) % p
        while i < m and t2 != 1:
            t2 = (t2 * t2) % p
            i += 1

        # Update
        b = pow(c, 1 << (m - i - 1), p)
        x = (x * b) % p
        bb = (b * b) % p
        t = (t * bb) % p
        c = bb
        m = i

    return x


def compute(m: int = TARGET_M) -> int:
    p = MOD
    exp_mod = p - 1  # Fermat exponent for F_p*

    inv2 = pow(2, -1, p)

    # sqrt(5) exists because p ≡ ±1 (mod 5)
    r = sqrt_mod_prime(5, p)

    def prepare_with_root(root: int):
        inv_r_local = pow(root, -1, p)
        phi = ((1 + root) * inv2) % p
        # g = phi^3
        g = (phi * phi) % p
        g = (g * phi) % p
        return phi, g, inv_r_local

    phi, g, inv_r = prepare_with_root(r)

    # Ensure we chose the root that matches the standard Fibonacci sign convention:
    # For n=3 (odd), F_3 should be 2.
    u = g
    uinv = pow(u, -1, p)
    f3 = ((u + uinv) % p) * inv_r % p
    if f3 != 2:
        r = (p - r) % p
        phi, g, inv_r = prepare_with_root(r)
        u = g
        uinv = pow(u, -1, p)
        f3 = ((u + uinv) % p) * inv_r % p
        if f3 != 2:
            raise RuntimeError("Failed to orient sqrt(5) consistently")

    inv32 = pow(32, -1, p)

    # Assert the example from the statement: s(0) = 33.
    # n=0 => k = 3*5^0 = 3, so this uses phi^3 directly.
    l3 = (u - uinv) % p
    # F_3^5 and L_3^5
    f3_2 = (f3 * f3) % p
    f3_4 = (f3_2 * f3_2) % p
    f3_5 = (f3_4 * f3) % p
    l3_2 = (l3 * l3) % p
    l3_4 = (l3_2 * l3_2) % p
    l3_5 = (l3_4 * l3) % p
    s0 = ((f3_5 + l3_5) % p) * inv32 % p
    assert s0 == 33

    # We need sum_{i=2..m} s(F_i).
    # Let E_i = 5^{F_i} (mod p-1). Then k = 3*5^{F_i} is odd, and
    # with u = phi^k = (phi^3)^{E_i} = g^{E_i}:
    #   F_k = (u + u^{-1}) / sqrt(5)
    #   L_k = (u - u^{-1})
    # and s(F_i) = (F_k^5 + L_k^5) / 32.

    total = 0

    # E_1 = 5^{F_1} = 5, E_2 = 5^{F_2} = 5
    e_im2 = 5 % exp_mod
    e_im1 = 5 % exp_mod

    pow_mod = pow
    mod = p
    base = g
    inv_sqrt5 = inv_r

    # Process i=2
    e = e_im1
    u = pow_mod(base, e, mod)
    uinv = pow_mod(u, -1, mod)
    f = ((u + uinv) % mod) * inv_sqrt5 % mod
    l = (u - uinv) % mod

    f2 = (f * f) % mod
    f4 = (f2 * f2) % mod
    f5 = (f4 * f) % mod
    l2 = (l * l) % mod
    l4 = (l2 * l2) % mod
    l5 = (l4 * l) % mod
    term = (f5 + l5) % mod
    term = (term * inv32) % mod
    total = term

    # Process i=3..m
    for _ in range(3, m + 1):
        e = (e_im1 * e_im2) % exp_mod
        e_im2, e_im1 = e_im1, e

        u = pow_mod(base, e, mod)
        uinv = pow_mod(u, -1, mod)

        f = ((u + uinv) % mod) * inv_sqrt5 % mod
        l = u - uinv
        if l < 0:
            l += mod

        f2 = (f * f) % mod
        f4 = (f2 * f2) % mod
        f5 = (f4 * f) % mod
        l2 = (l * l) % mod
        l4 = (l2 * l2) % mod
        l5 = (l4 * l) % mod

        term = f5 + l5
        term %= mod
        term = (term * inv32) % mod

        total += term
        if total >= mod:
            total -= mod

    return total


def main() -> None:
    import sys

    m = TARGET_M
    if len(sys.argv) >= 2:
        m = int(sys.argv[1])
    print(compute(m))


if __name__ == "__main__":
    main()
