#!/usr/bin/env python3
"""Project Euler 936: Peerless Trees

A peerless tree is a tree with no edge between two vertices of the same degree.
Let P(n) be the number of peerless trees on n unlabelled vertices.
Define S(N) = sum_{n=3..N} P(n).

This program computes S(50).

No external libraries are used.
"""

from __future__ import annotations


def _convolve(a: list[int], b: list[int], max_n: int) -> list[int]:
    """Truncated convolution for ordinary generating functions."""
    res = [0] * (max_n + 1)
    # a[0] and b[0] are always 0 here, but keep generic.
    for i, ai in enumerate(a):
        if ai == 0:
            continue
        lim = max_n - i
        for j in range(0, lim + 1):
            bj = b[j]
            if bj:
                res[i + j] += ai * bj
    return res


def _multiset_u_coeff_exact_size(A: list[int], m: int, max_d: int) -> list[int]:
    """Return the x-series for the coefficient of u^m in

        exp( sum_{k>=1} u^k/k * A(x^k) )

    where A is an ordinary generating function in x (A[0] must be 0).

    The return value is a list coeff[d] = [x^d] (u^m-coefficient).

    Uses the standard recurrence for exp-series coefficients:
      E_0 = 1
      E_n = (1/n) * sum_{k=1..n} A(x^k) * E_{n-k}

    computed coefficient-by-coefficient in x.
    """
    # E[n][d] = [x^d] coefficient of u^n.
    E = [[0] * (max_d + 1) for _ in range(m + 1)]
    E[0][0] = 1

    for d in range(1, max_d + 1):
        max_n = min(m, d)  # need at least n positive-size elements to make size d
        for n in range(1, max_n + 1):
            acc = 0
            for k in range(1, n + 1):
                # [x^d] A(x^k) * E[n-k]
                # A(x^k) has only terms x^{k*j} with coefficient A[j].
                dk = d // k
                Enk = E[n - k]
                for j in range(1, dk + 1):
                    aj = A[j]
                    if aj:
                        acc += aj * Enk[d - k * j]
            # Division is exact in this combinatorial setting.
            if acc % n != 0:
                raise ArithmeticError("non-integer multiset coefficient")
            E[n][d] = acc // n

    return E[m]


def _compute_planted_peerless(max_n: int) -> tuple[list[list[int]], list[int]]:
    """Compute planted peerless trees.

    g[r][s] = number of planted peerless trees with root outdegree r (children count)
              and total size s (number of vertices).

    total[s] = sum_r g[r][s]

    In a planted tree every vertex has a parent edge, so the peerless condition
    becomes: adjacent vertices must have different outdegrees.

    The root has outdegree r and its children are an unlabelled multiset of size r
    chosen from all planted trees except those whose root outdegree is also r.
    """
    max_d = max_n - 1  # child-size total

    g = [[0] * (max_n + 1) for _ in range(max_n)]  # r=0..max_n-1
    total = [0] * (max_n + 1)

    # e[t][n][d] = [x^d] coefficient of u^n in exp(sum u^k/k * A_t(x^k)),
    # where A_t = total - g[t] (children cannot have outdegree t).
    e: list[list[list[int]]] = [
        [[0] * (max_d + 1) for _ in range(t + 1)] for t in range(max_n)
    ]
    for t in range(max_n):
        e[t][0][0] = 1

    # Base planted tree: a single vertex with the parent edge and no children.
    g[0][1] = 1
    total[1] = 1

    for s in range(2, max_n + 1):
        d = s - 1

        # Compute e[*][*][d] using only sizes <= d.
        for t in range(max_n):
            max_children = min(t, d)
            for n in range(1, max_children + 1):
                acc = 0
                for k in range(1, n + 1):
                    dk = d // k
                    etnk = e[t][n - k]
                    for j in range(1, dk + 1):
                        aj = total[j] - g[t][j]
                        if aj:
                            acc += aj * etnk[d - k * j]
                if acc % n != 0:
                    raise ArithmeticError("non-integer planted coefficient")
                e[t][n][d] = acc // n

        # Now extract planted trees of size s: g[t][s] = [x^d] (u^t-coefficient).
        tot_s = 0
        for t in range(1, min(max_n, d + 1)):
            val = e[t][t][d]
            g[t][s] = val
            tot_s += val
        total[s] = tot_s

    return g, total


def _vertex_rooted_counts(
    max_n: int, g: list[list[int]], total: list[int]
) -> list[int]:
    """V[n] = number of peerless trees on n vertices rooted at a vertex."""
    max_d = max_n - 1
    V = [0] * (max_n + 1)

    for m in range(0, max_n):
        # Root degree is m, so we need an unlabelled multiset of exactly m planted trees.
        # A neighbour's outdegree cannot be m-1 (since neighbour degree = outdeg+1).
        banned = m - 1
        A = [0] * (max_d + 1)
        for sz in range(1, max_d + 1):
            v = total[sz]
            if banned >= 0:
                v -= g[banned][sz]
            A[sz] = v

        coeff = _multiset_u_coeff_exact_size(A, m, max_d)  # child total size -> count

        for child_size, c in enumerate(coeff):
            if c == 0:
                continue
            n = child_size + 1  # add root vertex
            if n <= max_n:
                V[n] += c

    return V


def _edge_rooted_counts(max_n: int, g: list[list[int]], total: list[int]) -> list[int]:
    """E[n] = number of peerless trees on n vertices rooted at an undirected edge."""
    # Oriented edge-rooted trees correspond to an ordered pair of planted trees
    # with different outdegrees (since endpoint degrees must differ).
    # So O = total^2 - sum_r (G_r^2).
    # For peerless trees, reversing the root edge is never an automorphism, so E = O/2.

    T = total[:]  # coefficient of x^s
    TT = _convolve(T, T, max_n)

    sum_sq = [0] * (max_n + 1)
    for r in range(max_n):
        sq = _convolve(g[r], g[r], max_n)
        for i in range(0, max_n + 1):
            sum_sq[i] += sq[i]

    oriented = [TT[i] - sum_sq[i] for i in range(max_n + 1)]

    E = [0] * (max_n + 1)
    for i, v in enumerate(oriented):
        if v % 2 != 0:
            raise ArithmeticError("expected even oriented-edge coefficient")
        E[i] = v // 2

    return E


def peerless_counts_upto(max_n: int) -> list[int]:
    """Return P[0..max_n], where P[n] is the number of peerless unlabelled trees on n vertices."""
    g, total = _compute_planted_peerless(max_n)
    V = _vertex_rooted_counts(max_n, g, total)
    E = _edge_rooted_counts(max_n, g, total)

    # Dissymmetry theorem for trees (unlabelled):
    #   U = V - O + E
    # Here O = 2E (no edge can be reversed by an automorphism), so U = V - E.
    P = [0] * (max_n + 1)
    for n in range(0, max_n + 1):
        P[n] = V[n] - E[n]

    return P


def main() -> None:
    N = 50
    P = peerless_counts_upto(N)

    # Test values from the problem statement.
    assert P[7] == 6

    S = [0] * (N + 1)
    running = 0
    for n in range(3, N + 1):
        running += P[n]
        S[n] = running

    assert S[10] == 74

    print(S[N])


if __name__ == "__main__":
    main()
