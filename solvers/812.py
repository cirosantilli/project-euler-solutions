#!/usr/bin/env python3
# Project Euler 812 - Dynamical Polynomials
# Computes S(10_000) mod 998244353, with sanity checks from the statement.
# No external libraries are used.

MOD = 998244353
PRIMITIVE_ROOT = 3


def _sieve_primes(n: int) -> list[int]:
    """Return primes <= n (simple sieve)."""
    if n < 2:
        return []
    bs = bytearray(b"\x01") * (n + 1)
    bs[0:2] = b"\x00\x00"
    p = 2
    while p * p <= n:
        if bs[p]:
            step = p
            start = p * p
            bs[start : n + 1 : step] = b"\x00" * (((n - start) // step) + 1)
        p += 1
    return [i for i in range(2, n + 1) if bs[i]]


def _ntt(a: list[int], invert: bool) -> None:
    """In-place iterative NTT over MOD."""
    n = len(a)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]

    length = 2
    mod = MOD
    g = PRIMITIVE_ROOT
    while length <= n:
        wlen = pow(g, (mod - 1) // length, mod)
        if invert:
            wlen = pow(wlen, mod - 2, mod)
        half = length >> 1
        for i in range(0, n, length):
            w = 1
            j_end = i + half
            j2 = i + half
            for j in range(i, j_end):
                u = a[j]
                v = (a[j2] * w) % mod
                a[j] = (u + v) % mod
                a[j2] = (u - v) % mod
                w = (w * wlen) % mod
                j2 += 1
        length <<= 1

    if invert:
        inv_n = pow(n, mod - 2, mod)
        for i in range(n):
            a[i] = (a[i] * inv_n) % mod


def _polymul(a: list[int], b: list[int]) -> list[int]:
    """Polynomial multiply modulo MOD."""
    if not a or not b:
        return []
    n = len(a) + len(b) - 1

    # Small-size fallback
    if min(len(a), len(b)) <= 32:
        res = [0] * n
        mod = MOD
        for i, ai in enumerate(a):
            if ai:
                for j, bj in enumerate(b):
                    res[i + j] = (res[i + j] + ai * bj) % mod
        return res

    size = 1
    while size < n:
        size <<= 1
    fa = a + [0] * (size - len(a))
    fb = b + [0] * (size - len(b))
    _ntt(fa, invert=False)
    _ntt(fb, invert=False)
    mod = MOD
    for i in range(size):
        fa[i] = (fa[i] * fb[i]) % mod
    _ntt(fa, invert=True)
    return fa[:n]


def _poly_der(a: list[int]) -> list[int]:
    mod = MOD
    return [(i * a[i]) % mod for i in range(1, len(a))]


def _poly_int(a: list[int], invs: list[int]) -> list[int]:
    mod = MOD
    res = [0] * (len(a) + 1)
    for i, x in enumerate(a, start=1):
        res[i] = (x * invs[i]) % mod
    return res


def _poly_inv(a: list[int], n: int) -> list[int]:
    """Inverse series of a modulo x^n. Requires a[0] != 0."""
    assert a and a[0] != 0
    mod = MOD
    res = [pow(a[0], mod - 2, mod)]
    m = 1
    while m < n:
        m2 = 2 * m
        if m2 > n:
            m2 = n
        t = _polymul(a[:m2], res)[:m2]
        t[0] = (2 - t[0]) % mod
        for i in range(1, m2):
            t[i] = (-t[i]) % mod
        res = _polymul(res, t)[:m2]
        m = m2
    return res[:n]


def _poly_ln(a: list[int], n: int, invs: list[int]) -> list[int]:
    """ln(a) mod x^n. Requires a[0] == 1."""
    assert a and a[0] == 1
    der = _poly_der(a)
    inv_a = _poly_inv(a, n)
    q = _polymul(der, inv_a)[: max(0, n - 1)]
    return _poly_int(q, invs)[:n]


def _poly_exp(f: list[int], n: int, invs: list[int]) -> list[int]:
    """exp(f) mod x^n. Requires f[0] == 0."""
    assert not f or f[0] == 0
    mod = MOD
    g = [1]
    m = 1
    while m < n:
        m2 = 2 * m
        if m2 > n:
            m2 = n
        g_pad = g + [0] * (m2 - len(g))
        ln_g = _poly_ln(g_pad, m2, invs)
        diff = [0] * m2
        for i in range(m2):
            fi = f[i] if i < len(f) else 0
            diff[i] = (fi - ln_g[i]) % mod
        diff[0] = (diff[0] + 1) % mod
        g = _polymul(g, diff)[:m2]
        m = m2
    return g[:n]


def _special_component(N: int) -> list[int]:
    """
    Generating series for the special orbit {2, -2, 0} (i.e. m = 1, 2, 4, 8, ...),
    incorporating the unique ramification at x=0.
    Returns coefficients up to x^N.
    """
    mod = MOD
    inv2 = (mod + 1) // 2

    # V1(x) = Π_{r>=1, 2^r<=N} (1 - x^{2^r})^{-1}
    v1 = [0] * (N + 1)
    v1[0] = 1
    p = 2
    while p <= N:
        for d in range(p, N + 1):
            v1[d] = (v1[d] + v1[d - p]) % mod
        p <<= 1

    # Vminus(x) = Π_{r>=1, 2^r<=N} (1 + x^{2^r})^{-1}
    vminus = [0] * (N + 1)
    vminus[0] = 1
    p = 2
    while p <= N:
        prev = vminus
        new = [0] * (N + 1)
        for d in range(N + 1):
            val = prev[d]
            if d >= p:
                # new[d] + new[d-p] = prev[d]
                val = (val - new[d - p]) % mod
            new[d] = val
        vminus = new
        p <<= 1

    # P(x) = 1/2 * ( (1+x)V1 + (1-x)Vminus )
    pser = [0] * (N + 1)
    for d in range(N + 1):
        t1 = v1[d] + (v1[d - 1] if d > 0 else 0)
        t2 = vminus[d] - (vminus[d - 1] if d > 0 else 0)
        pser[d] = ((t1 + t2) % mod) * inv2 % mod

    # Multiply by 1/(1-x): prefix sums
    a = [0] * (N + 1)
    s = 0
    for d in range(N + 1):
        s = (s + pser[d]) % mod
        a[d] = s

    # Multiply by 1/(1-x^2): b[d] = a[d] + b[d-2]
    b = [0] * (N + 1)
    for d in range(N + 1):
        val = a[d]
        if d >= 2:
            val = (val + b[d - 2]) % mod
        b[d] = val
    return b


def _add_component_multiplicities(c: list[int], N: int) -> None:
    """
    Populate c[w] with the number of odd m0>1 components contributing a part-size w.

    For odd m0>1, the component contributes parts of size
        W_t = Σ_{k=0..t} deg(P_{2^k m0})
    with one independent geometric factor per t.
    """
    limit_phi = 2 * N

    primes = [p for p in _sieve_primes(limit_phi + 1) if p & 1]  # odd primes only

    def dfs(start_idx: int, n_val: int, phi_val: int) -> None:
        # Record this odd m0 (exclude 1)
        if n_val > 1:
            ph = phi_val
            deg0 = ph // 2  # integer for all odd n_val>1 with phi<=limit_phi
            w = 0
            k = 0
            while True:
                if k == 0 or k == 1:
                    deg = deg0
                else:
                    deg = ph * (1 << (k - 2))
                w += deg
                if w > N:
                    break
                c[w] += 1
                k += 1

        # Extend by adding new prime powers
        for i in range(start_idx, len(primes)):
            p = primes[i]
            if phi_val * (p - 1) > limit_phi:
                break

            # exponent 1
            n2 = n_val * p
            phi2 = phi_val * (p - 1)
            dfs(i + 1, n2, phi2)

            # exponent >=2
            n_e = n2 * p
            phi_e = phi2 * p
            while phi_e <= limit_phi:
                dfs(i + 1, n_e, phi_e)
                n_e *= p
                phi_e *= p

    dfs(0, 1, 1)


def solve(N: int = 10_000) -> int:
    # Build multiplicities for "regular" odd components (odd m0>1)
    c = [0] * (N + 1)
    _add_component_multiplicities(c, N)

    # Build h(x) = log(F)(x) where F(x) = Π (1 - x^a)^(-c[a])
    g = [0] * (N + 1)
    mod = MOD
    for d in range(1, N + 1):
        cd = c[d]
        if cd:
            add = (d * (cd % mod)) % mod
            for k in range(d, N + 1, d):
                g[k] = (g[k] + add) % mod

    invs = [0] * (N + 2)
    for i in range(1, N + 2):
        invs[i] = pow(i, mod - 2, mod)

    h = [0] * (N + 1)
    for k in range(1, N + 1):
        h[k] = g[k] * invs[k] % mod

    colored = _poly_exp(h, N + 1, invs)

    # Special component (m = 1,2,4,8,...)
    special = _special_component(N)

    # Total generating function is convolution
    total = _polymul(colored, special)[: N + 1]

    # Asserts from the problem statement
    assert total[2] == 6
    assert total[5] == 58
    assert total[20] == 122087

    return total[N] % mod


if __name__ == "__main__":
    print(solve())
