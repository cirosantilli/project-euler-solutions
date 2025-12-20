#!/usr/bin/env python3
"""Project Euler 942: Mersenne's Square Root

We need R(q): the minimal x > 0 such that x^2 \equiv q (mod p) where p = 2^q - 1.
For the given q, p is a (Mersenne) prime.

Key identity (q an odd prime, q \equiv 1 (mod 4)):
Let (a/q) be the Legendre symbol. Because p = 2^q - 1 is prime, 2 has exact order q modulo p,
so 2 is a primitive q-th root of unity in F_p. The quadratic Gauss sum

    G = sum_{a=1..q-1} (a/q) * 2^a    (computed in F_p)

satisfies G^2 \equiv q (mod p). Thus G is a square root of q modulo p (up to sign).

We only need the final answer modulo MOD = 1e9+7, and G is a +/- sum of powers of two.
So we compute the same sum modulo MOD.

The minimal root is min(G, p-G) as integers in [0,p). For q \equiv 1 (mod 4), the top term
2^{q-1} has coefficient (q-1/q) = (−1/q) = +1, so:

    G = 2^{q-1} + L,   where L = sum_{a=1..q-2} (a/q) 2^a.

The sign of L is determined by its highest term 2^{q-2} with coefficient (−2/q).
If (−2/q) = +1 then L > 0 and G > 2^{q-1}, hence the minimal root is p-G.
If (−2/q) = −1 then L < 0 and the minimal root is G.

We compute the Legendre symbol table by marking quadratic residues r = i^2 mod q.
"""

import sys

MOD = 1_000_000_007


def gauss_sum_mod(q: int, mod: int) -> int:
    """Return G(q) = sum_{a=1..q-1} (a/q) * 2^a (mod mod), where q is an odd prime."""

    # Mark quadratic residues mod q.
    # Using i^2 mod q for i=1..(q-1)//2 gives each nonzero residue exactly once.
    qr = bytearray(q)  # qr[a] == 1 iff a is a nonzero quadratic residue mod q
    half = (q - 1) // 2

    sq = 1  # 1^2 mod q
    delta = 3  # difference between consecutive squares: (i+1)^2 - i^2 = 2i+1
    for _ in range(half):
        qr[sq] = 1
        sq += delta
        if sq >= q:
            sq -= q
        delta += 2

    # Accumulate +/- 2^a in increasing a, keeping everything reduced mod 'mod'.
    s = 0
    pow2 = 2 % mod  # 2^1

    # Local bindings for speed.
    qr_mv = qr
    MOD_local = mod
    for a in range(1, q):
        if qr_mv[a]:
            s += pow2
            if s >= MOD_local:
                s -= MOD_local
        else:
            s -= pow2
            if s < 0:
                s += MOD_local

        pow2 <<= 1
        if pow2 >= MOD_local:
            pow2 -= MOD_local

    return s


def legendre_minus_two_prime(q: int) -> int:
    """Return (−2/q) for an odd prime q as +1 or −1."""
    r = q & 7  # q % 8
    # (-2/q) = 1 for q ≡ 1,3 (mod 8); -1 for q ≡ 5,7 (mod 8)
    return 1 if (r == 1 or r == 3) else -1


def r_mod(q: int, mod: int = MOD) -> int:
    """Compute R(q) modulo 'mod' for q an odd prime with p=2^q-1 prime.

    This implementation is designed for the instance in the problem statement.
    """
    if q < 3 or (q & 1) == 0:
        raise ValueError("q must be an odd prime >= 3")

    g = gauss_sum_mod(q, mod)
    p_mod = (pow(2, q, mod) - 1) % mod

    # For the intended instance, q ≡ 1 (mod 4), hence G^2 ≡ q (mod p).
    # Decide whether the minimal root is G or p-G using the sign of (-2/q).
    if legendre_minus_two_prime(q) == 1:
        return (p_mod - g) % mod
    return g


def solve() -> None:
    # Test values from the problem statement.
    assert r_mod(5) == 6
    assert r_mod(17) == 47569

    q = 74_207_281
    print(r_mod(q))


if __name__ == "__main__":
    solve()
