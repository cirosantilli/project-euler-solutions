#!/usr/bin/env python3
"""Project Euler 840: Sum of Products

Computes S(5*10^4) mod 999676999.

No external libraries are used.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

MOD = 999_676_999
N = 50_000


# ----------------------- Number theory helpers -----------------------


def sieve_spf(limit: int) -> List[int]:
    """Return smallest prime factor for every number up to limit (inclusive)."""
    spf = list(range(limit + 1))
    spf[0] = 0
    if limit >= 1:
        spf[1] = 1
    r = int(limit**0.5)
    for i in range(2, r + 1):
        if spf[i] == i:  # prime
            step = i
            start = i * i
            for j in range(start, limit + 1, step):
                if spf[j] == j:
                    spf[j] = i
    return spf


def build_D(limit: int) -> List[int]:
    """Build D(n) for 1..limit.

    For n>1 this equals the arithmetic derivative n', computed via SPF:
      n = p*m (p prime) => n' = m + p*m'

    The problem defines D(1) = 1.
    """
    spf = sieve_spf(limit)
    d = [0] * (limit + 1)  # arithmetic derivative, d(1)=0 initially
    for n in range(2, limit + 1):
        p = spf[n]
        m = n // p
        d[n] = m + p * d[m]
    d[1] = 1
    return d


def build_inverses(limit: int, mod: int) -> List[int]:
    """Modular inverses for 1..limit (mod is prime)."""
    inv = [0] * (limit + 1)
    inv[1] = 1
    for i in range(2, limit + 1):
        inv[i] = (mod - (mod // i) * inv[mod % i] % mod) % mod
    return inv


def next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


# ------------------------------ NTT core ------------------------------

_REV_CACHE: Dict[int, List[int]] = {}


def _bitrev(n: int) -> List[int]:
    rev = _REV_CACHE.get(n)
    if rev is not None:
        return rev
    rev = [0] * n
    lg = n.bit_length() - 1
    for i in range(1, n):
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (lg - 1))
    _REV_CACHE[n] = rev
    return rev


def ntt(a: List[int], invert: bool, mod: int, primitive_root: int) -> None:
    """In-place iterative NTT (power-of-two length)."""
    n = len(a)
    rev = _bitrev(n)
    for i in range(n):
        j = rev[i]
        if i < j:
            a[i], a[j] = a[j], a[i]

    length = 2
    while length <= n:
        wlen = pow(primitive_root, (mod - 1) // length, mod)
        if invert:
            wlen = pow(wlen, mod - 2, mod)
        half = length >> 1
        for i in range(0, n, length):
            w = 1
            i_half = i + half
            for j in range(i, i_half):
                u = a[j]
                v = (a[j + half] * w) % mod
                x = u + v
                y = u - v
                a[j] = x if x < mod else x - mod
                a[j + half] = y if y >= 0 else y + mod
                w = (w * wlen) % mod
        length <<= 1

    if invert:
        inv_n = pow(n, mod - 2, mod)
        for i in range(n):
            a[i] = (a[i] * inv_n) % mod


# -------------------- Convolution with fixed B prefix --------------------


class NTTConvolver:
    """Convolve arbitrary A with fixed B-prefixes b[0:L] for power-of-two L.

    Convolution is done under three NTT-friendly primes, then combined with CRT
    and reduced modulo MOD.

    Cached: NTT(B) for each interval length L and each prime.
    """

    # NTT primes: p = c*2^k + 1, primitive root = 3 for these
    PRIMES: Tuple[Tuple[int, int], ...] = (
        (998_244_353, 3),
        (1_004_535_809, 3),
        (469_762_049, 3),
    )

    def __init__(self, b: List[int], mod_out: int):
        self.b = b
        self.mod_out = mod_out
        self._cache: Dict[int, Tuple[List[int], List[int], List[int]]] = {}

        (p1, _), (p2, _), (p3, _) = self.PRIMES
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

        self.inv_p1_mod_p2 = pow(p1, p2 - 2, p2)
        self.inv_p12_mod_p3 = pow((p1 * p2) % p3, p3 - 2, p3)

        self.p1_mod = p1 % mod_out
        self.p12_mod = (p1 * p2) % mod_out

    def _ensure_cache(self, L: int) -> Tuple[List[int], List[int], List[int]]:
        cached = self._cache.get(L)
        if cached is not None:
            return cached

        nfft = 2 * L
        out: List[List[int]] = []
        for p, root in self.PRIMES:
            fb = [0] * nfft
            # b is already modulo MOD, but we need modulo p
            for i in range(L):
                fb[i] = self.b[i] % p
            ntt(fb, False, p, root)
            out.append(fb)

        cached_t = (out[0], out[1], out[2])
        self._cache[L] = cached_t
        return cached_t

    def convolve_first_L(self, a_seg: List[int], L: int) -> List[int]:
        """Return first L coefficients of convolution a_seg * b[0:L] mod mod_out."""
        if not a_seg:
            return [0] * L

        nfft = 2 * L
        FB1, FB2, FB3 = self._ensure_cache(L)

        # Compute residues under each prime
        residues: List[List[int]] = []
        for (p, root), FB in (
            (self.PRIMES[0], FB1),
            (self.PRIMES[1], FB2),
            (self.PRIMES[2], FB3),
        ):
            fa = [0] * nfft
            # a_seg values are modulo mod_out; reduce into p
            for i, x in enumerate(a_seg):
                fa[i] = x % p
            ntt(fa, False, p, root)
            for i in range(nfft):
                fa[i] = (fa[i] * FB[i]) % p
            ntt(fa, True, p, root)
            residues.append(fa[:L])

        r1, r2, r3 = residues

        # CRT combine into mod_out using Garner-style reconstruction
        out = [0] * L
        p1, p2, p3 = self.p1, self.p2, self.p3
        inv_p1_mod_p2 = self.inv_p1_mod_p2
        inv_p12_mod_p3 = self.inv_p12_mod_p3
        mod_out = self.mod_out
        p1_mod = self.p1_mod
        p12_mod = self.p12_mod

        for i in range(L):
            a1 = r1[i]
            a2 = r2[i]
            a3 = r3[i]

            t1 = (a2 - a1) % p2
            t1 = (t1 * inv_p1_mod_p2) % p2
            x12 = a1 + p1 * t1  # exact mod p1*p2

            t2 = (a3 - (x12 % p3)) % p3
            t2 = (t2 * inv_p12_mod_p3) % p3

            out[i] = (
                a1 % mod_out + (p1_mod * (t1 % mod_out)) + (p12_mod * (t2 % mod_out))
            ) % mod_out

        return out


# ------------------------------ Main recurrence ------------------------------


def build_b(n: int, mod: int, D: List[int], size: int) -> List[int]:
    """Compute b[k] = sum_{m|k} m * D(m)^{k/m} (mod mod) for k<=n."""
    b = [0] * size
    for m in range(1, n + 1):
        wm = D[m] % mod
        mm = m % mod
        pwr = wm
        j = m
        while j <= n:
            b[j] = (b[j] + mm * pwr) % mod
            pwr = (pwr * wm) % mod
            j += m
    return b


def compute_G(n: int, mod: int) -> List[int]:
    """Compute G(0..n) modulo mod."""
    D = build_D(n)
    size = next_pow2(n + 1)
    b = build_b(n, mod, D, size)

    inv = build_inverses(n, mod)

    a = [0] * size  # a[k] = G(k)
    f = [0] * size  # f[t] = sum_{k=1..t} b[k]*a[t-k]

    THRESH = 256
    convolver = NTTConvolver(b, mod)

    def solve_block(l: int, r: int) -> None:
        rr = r if r <= n else n
        bj = b
        fj = f
        aj = a
        invj = inv
        md = mod
        for i in range(l, rr + 1):
            if i == 0:
                ai = 1
            else:
                ai = (fj[i] * invj[i]) % md
            aj[i] = ai
            max_k = rr - i
            if max_k <= 0 or ai == 0:
                continue
            # Update f[i+1..rr] by ai*b[k]
            for k in range(1, max_k + 1):
                fj[i + k] = (fj[i + k] + ai * bj[k]) % md

    def cdq(l: int, L: int) -> None:
        if l > n:
            return
        r = l + L - 1
        if L <= THRESH:
            solve_block(l, r)
            return

        half = L >> 1
        mid = l + half - 1

        cdq(l, half)

        if mid < n:
            a_seg = a[l : mid + 1]  # length = half
            conv = convolver.convolve_first_L(a_seg, L)
            rr = r if r <= n else n
            fj = f
            md = mod
            base = l
            for t in range(mid + 1, rr + 1):
                fj[t] = (fj[t] + conv[t - base]) % md

        cdq(mid + 1, half)

    cdq(0, size)
    return a[: n + 1]


def main() -> None:
    G = compute_G(N, MOD)

    # Tests from the problem statement
    assert G[10] == 164
    assert sum(G[1:11]) % MOD == 396

    print(sum(G[1 : N + 1]) % MOD)


if __name__ == "__main__":
    main()
