#!/usr/bin/env python3
"""
Project Euler 846: Magic Bracelets

Model:
- Vertices: allowed bead values <= N (1, 2, p^k, 2 p^k for odd prime p)
- Edge between distinct a,b if a*b = x^2 + 1 for some integer x.
- A bracelet corresponds to a simple cycle (length >= 3) in this undirected graph.
- Rotations and reflections are considered equivalent.

Key observations:
- If an odd prime p divides x^2+1 then -1 is a quadratic residue mod p, hence p ≡ 1 (mod 4).
  Therefore primes ≡ 3 (mod 4) never occur, so we can ignore them completely.
- x^2+1 has at most one factor of 2 (never divisible by 4), so there are no even-even edges.

Algorithm outline:
1) Generate allowed values and for each odd prime power p^k compute a square root of -1 mod p^k
   (Tonelli–Shanks for mod p, then Hensel lift to p^k).
2) For each odd node a:
     enumerate x satisfying x^2 ≡ -1 (mod a) with x^2+1 <= a*N,
     compute b = (x^2+1)/a and add edge if b is allowed and <= N.
3) To sum all bracelet potencies (sum of vertices on each cycle), we:
   - restrict to the induced subgraph on values <= N
   - prune to the 2-core (iteratively remove degree < 2 vertices)
   - decompose into biconnected components (Tarjan). Every cycle lies in exactly one block.
   - inside each block, enumerate all simple cycles once using a canonical rule
     (start at the minimum vertex in the cycle; break reflection symmetry by first<last rule).
   - if a block is a simple cycle (all degrees 2 and |E|=|V|), add it directly.

No external libraries used.
"""

from __future__ import annotations

import math
import sys
from collections import deque
from typing import Dict, List, Set, Tuple


def sieve_primes(n: int) -> List[int]:
    """Return list of primes <= n."""
    if n < 2:
        return []
    is_prime = bytearray(b"\x01") * (n + 1)
    is_prime[0:2] = b"\x00\x00"
    r = int(n**0.5)
    for i in range(2, r + 1):
        if is_prime[i]:
            start = i * i
            step = i
            is_prime[start : n + 1 : step] = b"\x00" * (((n - start) // step) + 1)
    return [i for i in range(2, n + 1) if is_prime[i]]


def tonelli_shanks(n: int, p: int) -> int:
    """
    Solve x^2 ≡ n (mod p) for odd prime p, assuming solution exists.
    Returns one solution x in [0, p-1].
    """
    n %= p
    if n == 0:
        return 0
    if p == 2:
        return n

    # Check residue
    if pow(n, (p - 1) // 2, p) != 1:
        raise ValueError("n is not a quadratic residue mod p")

    # Fast path
    if p % 4 == 3:
        return pow(n, (p + 1) // 4, p)

    # Write p-1 = q*2^s with q odd
    q = p - 1
    s = 0
    while q % 2 == 0:
        q //= 2
        s += 1

    # Find z a quadratic non-residue
    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1

    m = s
    c = pow(z, q, p)
    t = pow(n, q, p)
    x = pow(n, (q + 1) // 2, p)

    while t != 1:
        # Find smallest i in [1, m) with t^(2^i)=1
        i = 1
        t2i = (t * t) % p
        while i < m and t2i != 1:
            t2i = (t2i * t2i) % p
            i += 1

        b = pow(c, 1 << (m - i - 1), p)
        x = (x * b) % p
        t = (t * b * b) % p
        c = (b * b) % p
        m = i

    return x


def build_graph(max_n: int) -> Dict[int, Set[int]]:
    """
    Build adjacency dict for all allowed values <= max_n that have at least one edge.
    """
    primes = sieve_primes(max_n)
    primes_1mod4 = [p for p in primes if p % 4 == 1]

    allowed: Set[int] = {1, 2}
    odd_nodes: List[int] = [1]  # odd nodes drive all edges (no even-even edges)
    root_mod: Dict[int, int] = {}

    # Precompute roots of -1 mod p^k for all prime powers <= max_n (p≡1 mod4).
    for p in primes_1mod4:
        r = tonelli_shanks(p - 1, p)  # sqrt(-1) mod p
        r = min(r, p - r)

        pk = p
        cur_r = r  # root modulo pk

        while pk <= max_n:
            allowed.add(pk)
            odd_nodes.append(pk)
            root_mod[pk] = cur_r
            if 2 * pk <= max_n:
                allowed.add(2 * pk)

            pk_next = pk * p
            if pk_next > max_n:
                break

            # Hensel lift root to modulus p^(k+1):
            # Given cur_r^2 + 1 ≡ 0 (mod pk), find cur_r' = cur_r + delta*pk s.t. mod pk*p.
            t = ((cur_r * cur_r + 1) // pk) % p
            inv = pow((2 * cur_r) % p, -1, p)
            delta = (-t * inv) % p
            cur_r = cur_r + delta * pk
            pk = pk_next

    adj: Dict[int, Set[int]] = {}

    def add_edge(a: int, b: int) -> None:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    # Generate edges by enumerating x solutions for each odd a.
    for a in odd_nodes:
        if a == 1:
            lim = math.isqrt(max_n - 1)
            for x in range(1, lim + 1):
                b = x * x + 1
                if b in allowed and b != 1:
                    add_edge(1, b)
            continue

        r = root_mod[a]
        lim = math.isqrt(a * max_n - 1)

        for residue in (r, a - r):
            if residue > lim:
                continue
            for x in range(residue, lim + 1, a):
                b = (x * x + 1) // a
                if b != a and b <= max_n and b in allowed:
                    add_edge(a, b)

    return adj


def induced_subgraph(adj_full: Dict[int, Set[int]], n: int) -> Dict[int, Set[int]]:
    """Induced subgraph on vertices <= n (copies neighbor sets)."""
    adj: Dict[int, Set[int]] = {}
    for v, nbrs in adj_full.items():
        if v > n:
            continue
        filt = {u for u in nbrs if u <= n}
        if filt:
            adj[v] = filt
    return adj


def prune_to_2core(adj: Dict[int, Set[int]]) -> None:
    """In-place prune vertices of degree < 2 until stable (keeps only 2-core)."""
    deg = {v: len(adj[v]) for v in adj}
    q = deque([v for v, d in deg.items() if d < 2])
    removed: Set[int] = set()

    while q:
        v = q.popleft()
        if v in removed:
            continue
        removed.add(v)
        for u in list(adj.get(v, ())):
            adj[u].remove(v)
            deg[u] -= 1
            if deg[u] < 2 and u not in removed:
                q.append(u)
        adj[v].clear()

    # Drop emptied vertices
    for v in list(adj.keys()):
        if not adj[v]:
            del adj[v]


def biconnected_components(adj: Dict[int, Set[int]]) -> List[List[Tuple[int, int]]]:
    """
    Tarjan biconnected components for undirected graph.
    Returns list of components, each as a list of edges (u,v).
    """
    sys.setrecursionlimit(1_000_000)

    disc: Dict[int, int] = {}
    low: Dict[int, int] = {}
    parent: Dict[int, int] = {}
    edge_stack: List[Tuple[int, int]] = []
    comps: List[List[Tuple[int, int]]] = []
    time = 0

    def dfs(u: int) -> None:
        nonlocal time
        time += 1
        disc[u] = low[u] = time

        for v in adj[u]:
            if v not in disc:
                parent[v] = u
                edge_stack.append((u, v))
                dfs(v)
                low[u] = min(low[u], low[v])

                if low[v] >= disc[u]:
                    comp: List[Tuple[int, int]] = []
                    while True:
                        e = edge_stack.pop()
                        comp.append(e)
                        if e == (u, v):
                            break
                    comps.append(comp)
            elif parent.get(u) != v and disc[v] < disc[u]:
                low[u] = min(low[u], disc[v])
                edge_stack.append((u, v))

    for u in adj:
        if u not in disc:
            dfs(u)
            if edge_stack:
                comps.append(edge_stack[:])
                edge_stack.clear()

    return comps


def cycle_sum_in_block(block_adj: Dict[int, Set[int]]) -> int:
    """
    Sum potencies (sum of vertices) of all simple cycles in this block,
    counting each cycle once up to rotation+reflection.
    """
    nodes = sorted(block_adj.keys())
    nbrs = {v: sorted(block_adj[v]) for v in nodes}

    total = 0

    def dfs(
        start: int,
        cur: int,
        first: int,
        visited: Set[int],
        path_sum: int,
        path_len: int,
    ) -> None:
        nonlocal total
        for nxt in nbrs[cur]:
            if nxt == start:
                if path_len >= 3 and first < cur:
                    total += path_sum
                continue
            if nxt <= start or nxt in visited:
                continue
            visited.add(nxt)
            dfs(start, nxt, first, visited, path_sum + nxt, path_len + 1)
            visited.remove(nxt)

    for start in nodes:
        if len(nbrs[start]) < 2:
            continue
        for first in nbrs[start]:
            if first <= start:
                continue
            visited = {start, first}
            dfs(start, first, first, visited, start + first, 2)

    return total


def sum_potencies_all(adj: Dict[int, Set[int]]) -> int:
    """
    Sum potencies of all bracelets (cycles) in the graph.
    Mutates adj (prunes, etc.).
    """
    if not adj:
        return 0

    prune_to_2core(adj)
    if not adj:
        return 0

    comps = biconnected_components(adj)

    total = 0
    for comp_edges in comps:
        # Build block adjacency
        verts: Set[int] = set()
        block_adj: Dict[int, Set[int]] = {}

        for u, v in comp_edges:
            verts.add(u)
            verts.add(v)
            block_adj.setdefault(u, set()).add(v)
            block_adj.setdefault(v, set()).add(u)

        if len(verts) < 3:
            continue

        # Fast path: block is a simple cycle
        edge_count = len(comp_edges)
        if edge_count == len(verts) and all(len(block_adj[v]) == 2 for v in verts):
            total += sum(verts)
            continue

        total += cycle_sum_in_block(block_adj)

    return total


def F_from_full(adj_full: Dict[int, Set[int]], n: int) -> int:
    """Compute F(n) using a prebuilt full graph (copies the induced subgraph)."""
    adj = induced_subgraph(adj_full, n)
    return sum_potencies_all(adj)


def main() -> None:
    N = 10**6
    adj_full = build_graph(N)

    # Required checks from the problem statement:
    assert F_from_full(adj_full, 20) == 258
    assert F_from_full(adj_full, 100) == 538768

    # Print the requested answer (do not hardcode/ assert it).
    print(sum_potencies_all(adj_full))


if __name__ == "__main__":
    main()
