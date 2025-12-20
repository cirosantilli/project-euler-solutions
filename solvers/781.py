#!/usr/bin/env python3
"""
Project Euler 781 - Feynman Diagrams

Compute F(50_000) mod 1_000_000_007.

The problem provides:
  F(4) = 5
  F(8) = 319
These are asserted in main().
"""

MOD = 1_000_000_007

# Three NTT-friendly primes (pairwise coprime).
P1, G1 = 998244353, 3
P2, G2 = 1004535809, 3
P3, G3 = 469762049, 3


def _bitrev_permutation(n: int) -> list[int]:
    """Return bit-reversal permutation for length n (power of two)."""
    rev = [0] * n
    lg = n.bit_length() - 1
    for i in range(n):
        x = i
        r = 0
        for _ in range(lg):
            r = (r << 1) | (x & 1)
            x >>= 1
        rev[i] = r
    return rev


# Caches for speed.
_REV_CACHE: dict[int, list[int]] = {}


def _ntt(a: list[int], invert: bool, mod: int, root: int) -> None:
    """In-place iterative NTT."""
    n = len(a)
    rev = _REV_CACHE.get(n)
    if rev is None:
        rev = _bitrev_permutation(n)
        _REV_CACHE[n] = rev

    for i in range(n):
        j = rev[i]
        if i < j:
            a[i], a[j] = a[j], a[i]

    length = 2
    while length <= n:
        wlen = pow(root, (mod - 1) // length, mod)
        if invert:
            wlen = pow(wlen, mod - 2, mod)
        half = length >> 1
        for i in range(0, n, length):
            w = 1
            base = i
            for j in range(half):
                u = a[base + j]
                v = (a[base + j + half] * w) % mod
                x = u + v
                if x >= mod:
                    x -= mod
                y = u - v
                if y < 0:
                    y += mod
                a[base + j] = x
                a[base + j + half] = y
                w = (w * wlen) % mod
        length <<= 1

    if invert:
        n_inv = pow(n, mod - 2, mod)
        for i in range(n):
            a[i] = (a[i] * n_inv) % mod


def _convolution_prime(a: list[int], b: list[int], mod: int, root: int) -> list[int]:
    """Convolution under a single NTT prime modulus."""
    if not a or not b:
        return []
    need = len(a) + len(b) - 1
    n = 1
    while n < need:
        n <<= 1
    fa = a[:] + [0] * (n - len(a))
    fb = b[:] + [0] * (n - len(b))
    _ntt(fa, False, mod, root)
    _ntt(fb, False, mod, root)
    for i in range(n):
        fa[i] = (fa[i] * fb[i]) % mod
    _ntt(fa, True, mod, root)
    return fa[:need]


# CRT constants to reconstruct modulo MOD.
_INV_P1_MOD_P2 = pow(P1, -1, P2)
_P12_MOD_P3 = (P1 * P2) % P3
_INV_P12_MOD_P3 = pow(_P12_MOD_P3, -1, P3)

_P1_MOD_P3 = P1 % P3
_P1_MOD_MOD = P1 % MOD
_P12_MOD_MOD = (_P1_MOD_MOD * (P2 % MOD)) % MOD


def _crt3_to_mod(r1: int, r2: int, r3: int) -> int:
    """CRT from residues mod P1,P2,P3 directly to residue mod MOD."""
    # x = r1 + P1*t2 + P1*P2*t3 (unique mod P1*P2*P3)
    t2 = (r2 - (r1 % P2)) % P2
    t2 = (t2 * _INV_P1_MOD_P2) % P2

    x12_mod_p3 = (r1 + _P1_MOD_P3 * t2) % P3
    t3 = (r3 - x12_mod_p3) % P3
    t3 = (t3 * _INV_P12_MOD_P3) % P3

    return (r1 % MOD + _P1_MOD_MOD * (t2 % MOD) + _P12_MOD_MOD * (t3 % MOD)) % MOD


def poly_mul(a: list[int], b: list[int], limit: int) -> list[int]:
    """Multiply polynomials mod MOD, returning first 'limit' coefficients."""
    if not a or not b:
        return []
    need = min(limit, len(a) + len(b) - 1)
    a = a[: min(len(a), need)]
    b = b[: min(len(b), need)]

    # For P1, one subtraction suffices since MOD < 2*P1.
    a1 = [x if x < P1 else x - P1 for x in a]
    b1 = [x if x < P1 else x - P1 for x in b]

    # P2 > MOD, so already reduced.
    a2 = a[:]
    b2 = b[:]

    # MOD < 3*P3, so at most two subtractions.
    def _r3(x: int) -> int:
        if x >= P3:
            x -= P3
            if x >= P3:
                x -= P3
        return x

    a3 = [_r3(x) for x in a]
    b3 = [_r3(x) for x in b]

    c1 = _convolution_prime(a1, b1, P1, G1)
    c2 = _convolution_prime(a2, b2, P2, G2)
    c3 = _convolution_prime(a3, b3, P3, G3)

    res = [0] * need
    for i in range(need):
        res[i] = _crt3_to_mod(c1[i], c2[i], c3[i])
    return res


def poly_inv(a: list[int], n: int) -> list[int]:
    """Formal power series inverse of a (a[0] != 0) modulo x^n."""
    inv = [pow(a[0], MOD - 2, MOD)]
    m = 1
    while m < n:
        m2 = min(2 * m, n)
        t = poly_mul(a[:m2], inv, m2)  # t = a * inv (mod x^m2)
        u = [0] * m2
        u[0] = (2 - t[0]) % MOD
        for i in range(1, m2):
            u[i] = (-t[i]) % MOD
        inv = poly_mul(inv, u, m2)
        m = m2
    return inv


def build_series(M: int) -> tuple[list[int], list[int]]:
    """
    Build A(h) and B(h) for degrees 0..M:

      A_m = (2m-1)!! * [x^(2m)] e^{-x}/(1-x)
      B_m = (2m-1)!! * [x^(2m)] e^{-x}/(1-x)^2
    """
    N = 2 * M

    # invfact[k] = 1/k!
    fact = [1] * (N + 1)
    for i in range(1, N + 1):
        fact[i] = (fact[i - 1] * i) % MOD
    invfact = [1] * (N + 1)
    invfact[N] = pow(fact[N], MOD - 2, MOD)
    for i in range(N, 0, -1):
        invfact[i - 1] = (invfact[i] * i) % MOD

    # S[r] = sum_{k=0..r} (-1)^k / k!  (mod MOD)
    S = [0] * (N + 1)
    acc = 0
    for r in range(N + 1):
        term = invfact[r]
        if r & 1:
            term = MOD - term
        acc += term
        if acc >= MOD:
            acc -= MOD
        S[r] = acc

    A = [0] * (M + 1)
    B = [0] * (M + 1)

    df = 1  # (2m-1)!!, with (-1)!! = 1 at m=0
    for m in range(M + 1):
        if m > 0:
            df = (df * (2 * m - 1)) % MOD

        s2m = S[2 * m]
        s2m_1 = S[2 * m - 1] if 2 * m - 1 >= 0 else 0

        a = s2m
        b = ((2 * m + 1) * s2m + s2m_1) % MOD

        A[m] = (df * a) % MOD
        B[m] = (df * b) % MOD

    return A, B


def main() -> None:
    n = 50_000
    M = n // 2

    A, B = build_series(M)
    invA = poly_inv(A, M + 1)
    G = poly_mul(B, invA, M + 1)  # G[m] = F(2m) mod MOD

    # Problem statement checks:
    assert G[2] == 5, "Expected F(4)=5"
    assert G[4] == 319, "Expected F(8)=319"

    print(G[M] % MOD)


if __name__ == "__main__":
    main()
