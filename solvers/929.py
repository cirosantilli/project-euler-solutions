#!/usr/bin/env python3
"""
Project Euler 929: Odd-Run Compositions

Compute F(10^5) modulo 1111124111, where F(n) counts compositions of n
whose maximal runs of equal parts all have odd length.

No external libraries are used.
"""

import math

MOD = 1111124111
N = 100000

# FFT splitting base (15-bit)
_B = 1 << 15
_MASK = _B - 1
_B2 = _B * _B
_TAU = 2.0 * math.pi

# Caches for FFT
_rev_cache = {}
_roots_cache = {}


def _bitrev(n: int):
    """Bit-reversal permutation for power-of-two n."""
    rev = _rev_cache.get(n)
    if rev is not None:
        return rev
    logn = n.bit_length() - 1
    rev = [0] * n
    # O(n) construction
    for i in range(1, n):
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (logn - 1))
    _rev_cache[n] = rev
    return rev


def _get_roots(n: int):
    """
    Precompute roots for each stage length for both forward and inverse FFT.
    Returns (roots_fwd, roots_inv) as dicts: length -> list[complex] of size length//2.
    """
    cached = _roots_cache.get(n)
    if cached is not None:
        return cached

    roots_fwd = {}
    roots_inv = {}

    length = 2
    while length <= n:
        half = length >> 1
        ang = _TAU / length
        rf = [0j] * half
        ri = [0j] * half
        # Using cos/sin directly keeps numerical drift low for large transforms.
        for k in range(half):
            c = math.cos(ang * k)
            s = math.sin(ang * k)
            rf[k] = complex(c, s)
            ri[k] = complex(c, -s)  # conjugate of forward root
        roots_fwd[length] = rf
        roots_inv[length] = ri
        length <<= 1

    _roots_cache[n] = (roots_fwd, roots_inv)
    return roots_fwd, roots_inv


def _fft(a, invert: bool):
    """In-place iterative FFT over complex numbers. n must be a power of two."""
    n = len(a)
    rev = _bitrev(n)

    # Bit-reversal permutation
    for i in range(n):
        j = rev[i]
        if i < j:
            a[i], a[j] = a[j], a[i]

    roots_fwd, roots_inv = _get_roots(n)
    roots = roots_inv if invert else roots_fwd

    length = 2
    while length <= n:
        half = length >> 1
        rts = roots[length]
        for i0 in range(0, n, length):
            i = i0
            j = i0 + half
            for k in range(half):
                u = a[i]
                v = a[j] * rts[k]
                a[i] = u + v
                a[j] = u - v
                i += 1
                j += 1
        length <<= 1

    if invert:
        inv_n = 1.0 / n
        for i in range(n):
            a[i] *= inv_n


def _sround(x: float) -> int:
    """Round to nearest int, symmetric for negatives."""
    if x >= 0.0:
        return int(x + 0.5)
    return -int(-x + 0.5)


def _convolution_mod(a, b, limit=None):
    """
    Convolution (Cauchy product) of integer sequences a and b modulo MOD.
    Uses naive multiplication for small sizes; otherwise uses FFT with 15-bit splitting.

    Returns at most `limit` coefficients.
    """
    na = len(a)
    nb = len(b)
    if na == 0 or nb == 0:
        return []

    full_len = na + nb - 1
    if limit is None:
        limit = full_len
    else:
        limit = min(limit, full_len)

    # Small-case naive multiplication
    if na * nb <= 16384:
        res = [0] * limit
        for i, ai in enumerate(a):
            if ai == 0:
                continue
            maxj = min(nb, limit - i)
            for j in range(maxj):
                res[i + j] = (res[i + j] + ai * b[j]) % MOD
        return res

    n_fft = 1 << ((full_len - 1).bit_length())

    fa = [0j] * n_fft
    fb = [0j] * n_fft

    for i, x in enumerate(a):
        fa[i] = complex(x & _MASK, x >> 15)
    for i, x in enumerate(b):
        fb[i] = complex(x & _MASK, x >> 15)

    _fft(fa, False)
    _fft(fb, False)

    # Trick: with packed sequences (lo + i*hi), we need two inverse FFTs to recover
    # lo*lo, lo*hi, hi*lo, hi*hi convolutions.
    r = [0j] * n_fft
    r[0] = fa[0].conjugate() * fb[0]
    for i in range(1, n_fft):
        r[i] = fa[n_fft - i].conjugate() * fb[i]

    for i in range(n_fft):
        fa[i] *= fb[i]

    _fft(fa, True)
    _fft(r, True)

    res = [0] * limit
    for i in range(limit):
        pr = fa[i].real
        pi = fa[i].imag
        rr = r[i].real
        ri = r[i].imag

        c00 = _sround((pr + rr) * 0.5)
        c11 = _sround((rr - pr) * 0.5)
        c01 = _sround((pi + ri) * 0.5)
        c10 = _sround((pi - ri) * 0.5)

        res[i] = (c00 + (c01 + c10) * _B + c11 * _B2) % MOD

    return res


def _series_inverse(f, n_terms: int):
    """
    Invert a power series f(x) modulo x^n_terms, returning g with f*g = 1 mod x^n_terms.
    Uses Newton iteration with polynomial multiplications done modulo MOD.
    """
    if n_terms <= 0:
        return []
    f0 = f[0] % MOD
    if f0 == 0:
        raise ValueError("Series is not invertible: constant term is 0.")
    g = [pow(f0, MOD - 2, MOD)]  # MOD is prime

    m = 1
    while m < n_terms:
        m2 = min(2 * m, n_terms)

        fg = _convolution_mod(f[:m2], g, limit=m2)

        # h = 2 - fg  (mod MOD)
        for i in range(m2):
            fg[i] = (-fg[i]) % MOD
        fg[0] = (fg[0] + 2) % MOD

        g = _convolution_mod(g, fg, limit=m2)
        m = m2

    return g[:n_terms]


def solve():
    # Fibonacci numbers modulo MOD: F1=1, F2=1
    fib = [0] * (N + 1)
    if N >= 1:
        fib[1] = 1
    if N >= 2:
        fib[2] = 1
    for i in range(3, N + 1):
        x = fib[i - 1] + fib[i - 2]
        if x >= MOD:
            x -= MOD
        fib[i] = x

    # Lambert-series coefficient sieve:
    # s[n] = sum_{d|n} (-1)^(d-1) * Fib[d]  (mod MOD)
    s = [0] * (N + 1)
    for d in range(1, N + 1):
        val = fib[d]
        if (d & 1) == 0:
            val = MOD - val  # negate
        for k in range(d, N + 1, d):
            x = s[k] + val
            if x >= MOD:
                x -= MOD
            s[k] = x

    # We need T(x) = 1 / (1 - S(x)) where S(x) = sum_{n>=1} s[n] x^n.
    # So invert f(x) = 1 - S(x).
    f = [0] * (N + 1)
    f[0] = 1
    for i in range(1, N + 1):
        f[i] = (MOD - s[i]) % MOD

    t = _series_inverse(f, N + 1)

    # Example from the statement
    assert t[5] == 10

    print(t[N] % MOD)


if __name__ == "__main__":
    solve()
