#!/usr/bin/env python3
"""Project Euler 902: Permutation Powers

Compute P(m) and print P(100) modulo 1_000_000_007.

No third-party libraries are used.
"""

from __future__ import annotations

import bisect
import math

MOD = 1_000_000_007
A = 1_000_000_007  # multiplier in tau


def egcd(a: int, b: int) -> tuple[int, int, int]:
    """Return (g, x, y) such that a*x + b*y = g = gcd(a, b)."""
    x0, y0, x1, y1 = 1, 0, 0, 1
    while b:
        q = a // b
        a, b = b, a - q * b
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    return a, x0, y0


def inv_mod(a: int, m: int) -> int:
    """Modular inverse of a modulo m, assuming gcd(a,m)=1."""
    if m == 1:
        return 0
    g, x, _ = egcd(a % m, m)
    if g != 1:
        raise ValueError("inverse does not exist")
    return x % m


def rank_of_perm_small(p: list[int]) -> int:
    """Exact lexicographic rank (1-based) for a small permutation."""
    n = len(p)
    used = [False] * (n + 1)
    fact = [1] * (n + 1)
    for i in range(2, n + 1):
        fact[i] = fact[i - 1] * i
    r = 1
    for i in range(n):
        x = p[i]
        c = 0
        for v in range(1, x):
            if not used[v]:
                c += 1
        used[x] = True
        r += c * fact[n - 1 - i]
    return r


def lcm_upto(m: int) -> int:
    l = 1
    for k in range(2, m + 1):
        l = l // math.gcd(l, k) * k
    return l


def build_cycles(m: int, n: int) -> tuple[list[list[int]], list[int], list[int]]:
    """Build cycles of pi in forward order, plus element->(length, offset)."""
    cycles: list[list[int]] = [[] for _ in range(m + 1)]
    clen = [0] * (n + 1)
    coff = [0] * (n + 1)

    if n == 1:
        cycles[1] = [1]
        clen[1] = 1
        coff[1] = 0
        return cycles, clen, coff

    inva = inv_mod(A % n, n)

    def tau_inv(x: int) -> int:
        # x in 1..n. Solve (A*i mod n)+1 = x.
        t = (inva * ((x - 1) % n)) % n
        return n if t == 0 else t

    for l in range(1, m + 1):
        start = l * (l - 1) // 2 + 1
        cyc = [tau_inv(x) for x in range(start, start + l)]
        cycles[l] = cyc
        for idx, elem in enumerate(cyc):
            clen[elem] = l
            coff[elem] = idx

    return cycles, clen, coff


def precompute_factorials(n: int, mod: int) -> list[int]:
    fact = [1] * (n + 1)
    for i in range(2, n + 1):
        fact[i] = (fact[i - 1] * i) % mod
    return fact


def precompute_comp_and_scale(
    cycles: list[list[int]],
    m: int,
    lcm_all_mod: int,
) -> tuple[list[list[list[int]]], list[list[int]], list[list[int]]]:
    """Precompute:

    comp[a][b][d] = count over one period l=lcm(a,b) of times
                    pi^k on offset-pair difference d makes (b) < (a).
    gcd_tab[a][b] = gcd(a,b)
    scale[a][b]   = lcm_all / lcm(a,b)  (mod MOD)

    Here d is taken modulo gcd(a,b).
    """
    gcd_tab = [[0] * (m + 1) for _ in range(m + 1)]
    scale = [[0] * (m + 1) for _ in range(m + 1)]
    comp: list[list[list[int]]] = [[[] for _ in range(m + 1)] for __ in range(m + 1)]

    for a in range(1, m + 1):
        for b in range(1, m + 1):
            g = math.gcd(a, b)
            gcd_tab[a][b] = g
            l = (a // g) * b  # lcm(a,b)
            scale[a][b] = (lcm_all_mod * pow(l % MOD, MOD - 2, MOD)) % MOD

            Avals = cycles[a]
            Bvals = cycles[b]

            # Partition indices by residue mod g.
            A_subs = [Avals[r::g] for r in range(g)]
            B_sorted = [sorted(Bvals[r::g]) for r in range(g)]

            smallcounts = [0] * g
            for d in range(g):
                tot = 0
                for r in range(g):
                    s = (r - d) % g
                    Bs = B_sorted[s]
                    for aval in A_subs[r]:
                        tot += bisect.bisect_left(Bs, aval)
                # Over a full lcm-period the diagonal walk visits exactly l pairs.
                # Therefore this total is already the count over one period.
                smallcounts[d] = tot
            comp[a][b] = smallcounts

    return comp, gcd_tab, scale


def sum_ranks_over_full_period(m: int) -> tuple[int, int, int]:
    """Return (sumRanks mod MOD over k in [0..L-1], L_mod, m_fact_mod)."""
    n = m * (m + 1) // 2
    fact_n = precompute_factorials(n, MOD)

    # weights for Lehmer/inversion contribution: weight at position i is (n-i)!
    weights = [0] * (n + 1)
    for i in range(1, n + 1):
        weights[i] = fact_n[n - i]

    cycles, clen, coff = build_cycles(m, n)

    L = lcm_upto(m)
    L_mod = L % MOD

    comp, gcd_tab, scale = precompute_comp_and_scale(cycles, m, L_mod)

    # Sum of ranks over all exponents k mod L:
    # sum_k rank = L + sum_{i<j} (n-i)! * count_k[pi^k(j) < pi^k(i)].
    acc = 0
    mod = MOD

    # Local bindings for speed.
    wts = weights
    lens = clen
    offs = coff
    gcdt = gcd_tab
    scal = scale
    compm = comp

    THRESH = 1 << 62

    for i in range(1, n):
        wi = wts[i]
        ai = lens[i]
        offi = offs[i]
        for j in range(i + 1, n + 1):
            aj = lens[j]
            g = gcdt[ai][aj]
            if g == 1:
                d = 0
            else:
                d = (offi - offs[j]) % g
            small = compm[ai][aj][d]
            if small:
                cnt = (scal[ai][aj] * small) % mod
                acc += wi * cnt
                if acc >= THRESH:
                    acc %= mod

    acc %= mod
    sum_ranks_mod = (L_mod + acc) % mod

    # m! modulo MOD
    m_fact = 1
    for k in range(2, m + 1):
        m_fact = (m_fact * k) % mod

    return sum_ranks_mod, L_mod, m_fact


def P(m: int) -> int:
    """Compute P(m) modulo MOD."""
    sum_period, L_mod, m_fact = sum_ranks_over_full_period(m)
    return (sum_period * m_fact % MOD) * pow(L_mod, MOD - 2, MOD) % MOD


def main() -> None:
    # Test values from the problem statement.
    assert rank_of_perm_small([2, 1, 3]) == 3
    assert P(2) == 4
    assert P(3) == 780
    assert P(4) == 38810300

    print(P(100))


if __name__ == "__main__":
    main()
