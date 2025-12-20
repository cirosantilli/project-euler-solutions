#!/usr/bin/env python3
"""
Project Euler 767 — Window into a Matrix II

Counts 16×n binary matrices where every 2×k window has sum exactly k,
and prints B(10^5, 10^16) modulo 1_000_000_007.

No external libraries are used.
"""

MOD = 1_000_000_007

# Three NTT-friendly primes for exact integer convolution reconstruction.
# Their product is > 1e23, enough to reconstruct coefficients exactly for our bounds.
PRIMES = (
    (998244353, 3),
    (1004535809, 3),
    (469762049, 3),
)


def _ceil_pow2(x: int) -> int:
    n = 1
    while n < x:
        n <<= 1
    return n


def _prepare_rev(n: int) -> list[int]:
    """Bit-reversal permutation indices for n = power of two."""
    rev = [0] * n
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        rev[i] = j
    return rev


def _prepare_roots(
    n: int, mod: int, primitive_root: int
) -> tuple[list[int], list[int]]:
    """
    Precompute roots[i] = w^i for w = primitive n-th root,
    and roots_inv similarly for inverse transform.
    """
    w = pow(primitive_root, (mod - 1) // n, mod)
    roots = [1] * n
    for i in range(1, n):
        roots[i] = (roots[i - 1] * w) % mod

    w_inv = pow(w, mod - 2, mod)
    roots_inv = [1] * n
    for i in range(1, n):
        roots_inv[i] = (roots_inv[i - 1] * w_inv) % mod

    return roots, roots_inv


def _ntt(
    a: list[int],
    mod: int,
    roots: list[int],
    roots_inv: list[int],
    rev: list[int],
    invert: bool,
) -> None:
    """In-place iterative NTT."""
    n = len(a)

    # Bit-reversal permutation
    for i in range(n):
        j = rev[i]
        if i < j:
            a[i], a[j] = a[j], a[i]

    length = 2
    if not invert:
        while length <= n:
            half = length >> 1
            step = n // length
            for i0 in range(0, n, length):
                idx = 0
                for j in range(i0, i0 + half):
                    u = a[j]
                    v = (a[j + half] * roots[idx]) % mod
                    x = u + v
                    if x >= mod:
                        x -= mod
                    y = u - v
                    if y < 0:
                        y += mod
                    a[j] = x
                    a[j + half] = y
                    idx += step
            length <<= 1
    else:
        while length <= n:
            half = length >> 1
            step = n // length
            for i0 in range(0, n, length):
                idx = 0
                for j in range(i0, i0 + half):
                    u = a[j]
                    v = (a[j + half] * roots_inv[idx]) % mod
                    x = u + v
                    if x >= mod:
                        x -= mod
                    y = u - v
                    if y < 0:
                        y += mod
                    a[j] = x
                    a[j + half] = y
                    idx += step
            length <<= 1

        n_inv = pow(n, mod - 2, mod)
        for i in range(n):
            a[i] = (a[i] * n_inv) % mod


class _ConvolutionContext:
    """
    Holds NTT tables and CRT constants for a fixed transform length n (power of two).
    """

    __slots__ = ("n", "rev", "tables", "crt")

    def __init__(self, n: int):
        self.n = n
        self.rev = _prepare_rev(n)

        self.tables = []
        for p, g in PRIMES:
            roots, roots_inv = _prepare_roots(n, p, g)
            self.tables.append((p, roots, roots_inv))

        # CRT / Garner constants
        (p1, _), (p2, _), (p3, _) = PRIMES
        inv_p1_mod_p2 = pow(p1, p2 - 2, p2)

        p1_mod_p3 = p1 % p3
        p12_mod_p3 = (p1 * p2) % p3
        inv_p12_mod_p3 = pow(p12_mod_p3, p3 - 2, p3)

        self.crt = (
            p1,
            p2,
            p3,
            inv_p1_mod_p2,
            p1_mod_p3,
            p12_mod_p3,
            inv_p12_mod_p3,
            p1 % MOD,
            (p1 * p2) % MOD,
        )

    def convolution_mod(self, a: list[int], b: list[int]) -> list[int]:
        """Convolution in Z, reconstructed exactly via CRT, then reduced mod MOD."""
        need = len(a) + len(b) - 1
        n = self.n
        rev = self.rev

        convs: list[list[int]] = []
        for p, roots, roots_inv in self.tables:
            fa = [0] * n
            fb = [0] * n
            for i, x in enumerate(a):
                fa[i] = x % p
            for i, x in enumerate(b):
                fb[i] = x % p

            _ntt(fa, p, roots, roots_inv, rev, invert=False)
            _ntt(fb, p, roots, roots_inv, rev, invert=False)
            for i in range(n):
                fa[i] = (fa[i] * fb[i]) % p
            _ntt(fa, p, roots, roots_inv, rev, invert=True)

            convs.append(fa[:need])

        return self._crt_reduce(convs, need)

    def convolution_square_mod(self, a: list[int]) -> list[int]:
        """Convolution a*a, with one forward NTT per prime (then squaring pointwise)."""
        need = 2 * len(a) - 1
        n = self.n
        rev = self.rev

        convs: list[list[int]] = []
        for p, roots, roots_inv in self.tables:
            fa = [0] * n
            for i, x in enumerate(a):
                fa[i] = x % p

            _ntt(fa, p, roots, roots_inv, rev, invert=False)
            for i in range(n):
                fa[i] = (fa[i] * fa[i]) % p
            _ntt(fa, p, roots, roots_inv, rev, invert=True)

            convs.append(fa[:need])

        return self._crt_reduce(convs, need)

    def _crt_reduce(self, convs: list[list[int]], need: int) -> list[int]:
        """
        Combine residues modulo the three primes, reconstructing the exact integer
        coefficient (it fits under p1*p2*p3 for our bounds), then reduce mod MOD.
        """
        (
            p1,
            p2,
            p3,
            inv_p1_mod_p2,
            p1_mod_p3,
            p12_mod_p3,
            inv_p12_mod_p3,
            p1_mod_mod,
            p12_mod_mod,
        ) = self.crt

        out = [0] * need
        for i in range(need):
            r1 = convs[0][i]
            r2 = convs[1][i]
            r3 = convs[2][i]

            # Garner steps
            t1 = ((r2 - r1) % p2) * inv_p1_mod_p2 % p2
            x2_mod_p3 = (r1 + (p1_mod_p3 * t1) % p3) % p3
            t2 = ((r3 - x2_mod_p3) % p3) * inv_p12_mod_p3 % p3

            out[i] = (r1 + p1_mod_mod * (t1 % MOD) + p12_mod_mod * (t2 % MOD)) % MOD

        return out


def _pow16_mod(x: int) -> int:
    """x^16 (mod MOD) with fixed exponent using squaring."""
    x2 = (x * x) % MOD
    x4 = (x2 * x2) % MOD
    x8 = (x4 * x4) % MOD
    return (x8 * x8) % MOD


def B(k: int, n: int) -> int:
    """
    Compute B(k, n) modulo MOD.

    This implementation assumes n is divisible by k (true for the given data).
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if n % k != 0:
        raise ValueError("This implementation assumes n is divisible by k.")

    m = n // k
    A = pow(2, m, MOD)  # weight contributed by the special alternating column type

    # Factorials up to k
    fact = [1] * (k + 1)
    for i in range(1, k + 1):
        fact[i] = (fact[i - 1] * i) % MOD

    inv_fact = [1] * (k + 1)
    inv_fact[k] = pow(fact[k], MOD - 2, MOD)
    for i in range(k, 0, -1):
        inv_fact[i - 1] = (inv_fact[i] * i) % MOD

    fact16 = [0] * (k + 1)
    inv_fact16 = [0] * (k + 1)
    for i in range(k + 1):
        fact16[i] = _pow16_mod(fact[i])
        inv_fact16[i] = _pow16_mod(inv_fact[i])

    # NTT context sized for length 2*(k+1)-1
    need = 2 * (k + 1) - 1
    ntt_len = _ceil_pow2(need)
    ctx = _ConvolutionContext(ntt_len)

    # Franel-like numbers: f[r] = sum_x C(r,x)^16
    # Using: C(r,x)^16 = r!^16 / (x!^16 (r-x)!^16), so
    # f[r] = r!^16 * [y^r] (sum_n y^n / n!^16)^2
    a = inv_fact16[:]  # a[n] = 1/n!^16  (mod MOD)
    conv_aa = ctx.convolution_square_mod(a)  # length 2k+1

    f = [0] * (k + 1)
    for r in range(k + 1):
        f[r] = (fact16[r] * conv_aa[r]) % MOD

    # Binomial transform with parameter -2:
    # S[L] = sum_{r<=L} C(L,r) (-2)^{L-r} f[r]
    # Compute all S via factorial-convolution:
    # S[L]/L! = sum_{r<=L} (f[r]/r!) * ((-2)^{L-r}/(L-r)!)
    fprime = [0] * (k + 1)
    for r in range(k + 1):
        fprime[r] = (f[r] * inv_fact[r]) % MOD

    neg2 = MOD - 2
    b = [0] * (k + 1)
    p = 1
    for j in range(k + 1):
        b[j] = (p * inv_fact[j]) % MOD
        p = (p * neg2) % MOD

    conv_fb = ctx.convolution_mod(fprime, b)  # length 2k+1

    S = [0] * (k + 1)
    for L in range(k + 1):
        S[L] = (conv_fb[L] * fact[L]) % MOD

    # Final sum:
    # B(k,n) = sum_{a=0..k} C(k,a) * A^a * S[k-a]
    powA = [1] * (k + 1)
    for i in range(1, k + 1):
        powA[i] = (powA[i - 1] * A) % MOD

    fk = fact[k]
    ans = 0
    for a_cnt in range(k + 1):
        comb = fk * inv_fact[a_cnt] % MOD * inv_fact[k - a_cnt] % MOD
        ans = (ans + comb * powA[a_cnt] % MOD * S[k - a_cnt]) % MOD

    return ans


def _self_test() -> None:
    # Test values from the problem statement
    assert B(2, 4) == 65550
    assert B(3, 9) == 87273560  # given modulo 1_000_000_007


def main() -> None:
    _self_test()
    print(B(10**5, 10**16))


if __name__ == "__main__":
    main()
