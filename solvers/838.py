#!/usr/bin/env python3
"""
Project Euler 838 - Not Coprime

Let f(N) be the smallest positive integer that is not coprime to any positive
integer n <= N whose least significant digit is 3.

This program prints ln(f(10^6)) rounded to 6 digits after the decimal point.

Constraints:
- No external libraries (standard library only).
- Includes asserts for the test values given in the problem statement.
- Does NOT hardcode/print any known final answer inside the file.
"""

from __future__ import annotations

import bisect
import math
from collections import deque
from typing import Deque, Iterable, List, Set, Tuple


def sieve_primes(n: int) -> List[int]:
    """Sieve of Eratosthenes, returns all primes <= n."""
    if n < 2:
        return []
    is_prime = bytearray(b"\x01") * (n + 1)
    is_prime[0:2] = b"\x00\x00"
    r = int(n**0.5)
    for p in range(2, r + 1):
        if is_prime[p]:
            start = p * p
            step = p
            is_prime[start : n + 1 : step] = b"\x00" * (((n - start) // step) + 1)
    return [i for i in range(2, n + 1) if is_prime[i]]


def iroot3_floor(n: int) -> int:
    """Floor integer cube root."""
    x = int(round(n ** (1.0 / 3.0)))
    # Correct potential rounding drift
    while (x + 1) ** 3 <= n:
        x += 1
    while x**3 > n:
        x -= 1
    return x


def kahan_sum(values: Iterable[float]) -> float:
    """Kahan summation to reduce floating error for large sums."""
    s = 0.0
    c = 0.0
    for x in values:
        y = x - c
        t = s + y
        c = (t - s) - y
        s = t
    return s


class Dinic:
    """Dinic's max-flow algorithm for integer capacities."""

    def __init__(self, n: int):
        self.n = n
        # Each edge: [to, cap, rev_index]
        self.g: List[List[List[int]]] = [[] for _ in range(n)]

    def add_edge(self, u: int, v: int, cap: int) -> None:
        fwd = [v, cap, len(self.g[v])]
        rev = [u, 0, len(self.g[u])]
        self.g[u].append(fwd)
        self.g[v].append(rev)

    def max_flow(self, s: int, t: int) -> int:
        flow = 0
        n = self.n
        level = [-1] * n

        def bfs() -> bool:
            for i in range(n):
                level[i] = -1
            q: Deque[int] = deque([s])
            level[s] = 0
            while q:
                u = q.popleft()
                for v, cap, _ in self.g[u]:
                    if cap > 0 and level[v] < 0:
                        level[v] = level[u] + 1
                        q.append(v)
            return level[t] >= 0

        it = [0] * n

        def dfs(u: int, f: int) -> int:
            if u == t:
                return f
            for i in range(it[u], len(self.g[u])):
                it[u] = i
                e = self.g[u][i]
                v, cap, rev = e
                if cap > 0 and level[v] == level[u] + 1:
                    pushed = dfs(v, f if f < cap else cap)
                    if pushed:
                        e[1] -= pushed
                        self.g[v][rev][1] += pushed
                        return pushed
            return 0

        INF_FLOW = 10**30
        while bfs():
            for i in range(n):
                it[i] = 0
            while True:
                pushed = dfs(s, INF_FLOW)
                if not pushed:
                    break
                flow += pushed
        return flow

    def reachable_from(self, s: int) -> List[bool]:
        """Nodes reachable from s in residual graph (cap > 0)."""
        vis = [False] * self.n
        q: Deque[int] = deque([s])
        vis[s] = True
        while q:
            u = q.popleft()
            for v, cap, _ in self.g[u]:
                if cap > 0 and not vis[v]:
                    vis[v] = True
                    q.append(v)
        return vis


def forced_primes(primes: List[int], N: int) -> Set[int]:
    """
    Forced primes:
    - Every prime p <= N with p % 10 == 3 (since n = p is in the required set).
    - Every prime p <= cbrt(N) with p % 10 == 7, because p^3 ends with 3 and
      must share a factor with f(N), forcing p.
    """
    forced: Set[int] = set()
    forced.update(p for p in primes if p % 10 == 3)

    c = iroot3_floor(N)
    # Only primes == 7 mod 10 can have a power ending in 3; the smallest is p^3.
    forced.update(p for p in primes if p % 10 == 7 and p <= c)
    return forced


def build_prefix_bipartite(
    primes: List[int], N: int, forced: Set[int]
) -> Tuple[List[int], List[int], List[int]]:
    """
    Build the bipartite graph that remains after forcing.

    Left side: primes p % 10 == 7, not forced, and p*19 <= N (since smallest prime == 9 mod 10 is 19).
    Right side: primes q % 10 == 9 up to N // min_left.

    Edge (p,q) exists iff p*q <= N.
    For each p, the neighbors are exactly the prefix of right primes with q <= N//p.
    We store that as prefix lengths.
    """
    left = [p for p in primes if (p % 10 == 7) and (p not in forced) and (p * 19 <= N)]
    left.sort()
    if not left:
        return [], [], []

    pmin = left[0]
    right = [q for q in primes if (q % 10 == 9) and (q <= N // pmin)]
    right.sort()

    pref_len = []
    for p in left:
        lim = N // p
        pref_len.append(bisect.bisect_right(right, lim))
    return left, right, pref_len


def min_weight_vertex_cover_prefix_bipartite(
    left: List[int], right: List[int], pref_len: List[int]
) -> Tuple[Set[int], Set[int]]:
    """
    Minimum weight vertex cover in bipartite graph via min s-t cut.

    Weights are ln(p). To make capacities integral and stable, scale by 1e12
    for the flow computation, then recover the chosen vertices from the min-cut.
    """
    SCALE = 10**12

    wL = [int(round(math.log(p) * SCALE)) for p in left]
    wR = [int(round(math.log(q) * SCALE)) for q in right]

    total = sum(wL) + sum(wR)
    INF = total + 1  # must dominate any finite cut

    nL = len(left)
    nR = len(right)
    S = 0
    offL = 1
    offR = 1 + nL
    T = 1 + nL + nR
    din = Dinic(T + 1)

    for i, w in enumerate(wL):
        din.add_edge(S, offL + i, w)
    for j, w in enumerate(wR):
        din.add_edge(offR + j, T, w)

    # infinite edges L -> R for existing graph edges
    for i in range(nL):
        u = offL + i
        k = pref_len[i]
        for j in range(k):
            din.add_edge(u, offR + j, INF)

    din.max_flow(S, T)
    vis = din.reachable_from(S)

    # Standard recovery:
    # min vertex cover = (Left not reachable) U (Right reachable)
    coverL: Set[int] = set()
    coverR: Set[int] = set()
    for i, p in enumerate(left):
        if not vis[offL + i]:
            coverL.add(p)
    for j, q in enumerate(right):
        if vis[offR + j]:
            coverR.add(q)

    return coverL, coverR


def ln_f(N: int, return_primes: bool = False):
    primes = sieve_primes(N)
    forced = forced_primes(primes, N)

    left, right, pref_len = build_prefix_bipartite(primes, N, forced)
    coverL: Set[int] = set()
    coverR: Set[int] = set()
    if left and right:
        coverL, coverR = min_weight_vertex_cover_prefix_bipartite(left, right, pref_len)

    chosen = forced | coverL | coverR

    # Deterministic order helps Kahan summation stability a bit
    ln_total = kahan_sum(math.log(p) for p in sorted(chosen))

    if return_primes:
        return ln_total, chosen
    return ln_total


def _self_test() -> None:
    # Given: f(40) = 897 = 3*13*23 and ln f(40) rounds to 6.799056
    ln40, primes40 = ln_f(40, return_primes=True)
    prod = 1
    for p in primes40:
        prod *= p
    assert prod == 897
    assert f"{ln40:.6f}" == "6.799056"

    # Given: ln f(2800) ≈ 715.019337 (rounded to 6 digits)
    ln2800 = ln_f(2800)
    assert f"{ln2800:.6f}" == "715.019337"


def main() -> None:
    _self_test()
    N = 10**6
    ans = ln_f(N)
    print(f"{ans:.6f}")


if __name__ == "__main__":
    main()
