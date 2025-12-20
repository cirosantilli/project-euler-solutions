#!/usr/bin/env python3
"""Project Euler 857: Beautiful Graphs.

Prints G(10^7) mod (10^9+7) and asserts the sample values from the statement.

No external libraries are used.
"""

MOD = 1_000_000_007
TARGET_N = 10_000_000


def count_no_mono_triangle(k: int) -> int:
    """Count 2-colourings of edges of K_k with no monochromatic triangle."""
    if k <= 1:
        return 1

    edge_index = {}
    idx = 0
    for i in range(k):
        for j in range(i + 1, k):
            edge_index[(i, j)] = idx
            idx += 1

    triangles = []
    for i in range(k):
        for j in range(i + 1, k):
            for l in range(j + 1, k):
                triangles.append(
                    (
                        edge_index[(i, j)],
                        edge_index[(i, l)],
                        edge_index[(j, l)],
                    )
                )

    m = idx
    good = 0
    for mask in range(1 << m):
        ok = True
        for a, b, c in triangles:
            x = (mask >> a) & 1
            y = (mask >> b) & 1
            if x == y and x == ((mask >> c) & 1):
                ok = False
                break
        if ok:
            good += 1
    return good


def build_block_counts():
    """a[s] = valid green/brown edge-colourings inside a block of size s."""
    a = [0] * 6
    for s in range(1, 6):
        a[s] = count_no_mono_triangle(s)
    return a


def G_exact(n: int, a) -> int:
    """Exact integer G(n) for small n (used for asserts)."""
    fact = [1] * (n + 1)
    for i in range(2, n + 1):
        fact[i] = fact[i - 1] * i

    def comb(nn: int, kk: int) -> int:
        return fact[nn] // (fact[kk] * fact[nn - kk])

    g = [0] * (n + 1)
    g[0] = 1
    for m in range(1, n + 1):
        total = 0
        upto = 5 if m >= 5 else m
        for s in range(1, upto + 1):
            total += comb(m, s) * a[s] * g[m - s]
        g[m] = total
    return g[n]


def G_mod(n: int, a) -> int:
    """Compute G(n) modulo MOD for large n."""
    if n == 0:
        return 1

    # Coefficients for the ordinary generating function:
    # f_n = sum_s (a_s / s!) f_{n-s}
    coeff = [0] * 6
    fact_small = 1
    for s in range(1, 6):
        fact_small *= s
        coeff[s] = (a[s] * pow(fact_small, MOD - 2, MOD)) % MOD

    c1, c2, c3, c4, c5 = coeff[1], coeff[2], coeff[3], coeff[4], coeff[5]

    # Rolling recurrence for f_n, keeping the last 5 values.
    f1 = 1  # f_0
    f2 = f3 = f4 = f5 = 0

    fact = 1  # n! mod MOD
    for i in range(1, n + 1):
        fn = (c1 * f1 + c2 * f2 + c3 * f3 + c4 * f4 + c5 * f5) % MOD
        f5, f4, f3, f2, f1 = f4, f3, f2, f1, fn
        fact = (fact * i) % MOD

    return (fact * f1) % MOD


def main():
    a = build_block_counts()

    # Values provided in the problem statement
    assert G_exact(3, a) == 24
    assert G_exact(4, a) == 186
    assert G_exact(15, a) == 12472315010483328

    print(G_mod(TARGET_N, a))


if __name__ == "__main__":
    main()
