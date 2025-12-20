#!/usr/bin/env python3
"""
Project Euler 703: Circular Logic II

We work with B = {false, true} and B^n (all boolean strings of length n).
The problem defines a function f : B^n -> B^n that shifts left and appends
a new last bit computed from the first three bits:

  c_i = b_{i+1} for 1 <= i < n
  c_n = b1 AND (b2 XOR b3)

Let S(n) be the number of functions T : B^n -> B such that for all x in B^n:
  T(x) AND T(f(x)) = false

Equivalently, if we mark vertices x with T(x)=true, no directed edge x -> f(x)
may have both endpoints marked true. Since every vertex has exactly one outgoing
edge, the graph defined by f is a functional graph (cycles with in-trees).
Counting such T is exactly counting independent sets in the underlying graph.

This script computes S(20) modulo 1_001_001_011.
"""

from array import array
from collections import deque

MOD = 1001001011


def _build_successor_and_indegree(n: int):
    """Return (succ, indeg) arrays for the functional graph on 2^n states.

    State encoding:
      - We encode b1..bn as an integer s with b1 as the highest bit (bit n-1)
        and bn as the lowest bit (bit 0).
      - f shifts left (drops b1, moves b2 to b1, ..., bn to b_{n-1})
        and appends newbit as the new bn.
    """
    N = 1 << n
    succ = array("I", [0]) * N
    indeg = array("I", [0]) * N

    # After dropping b1 (highest bit), (s & mask) holds b2..bn.
    mask = (1 << (n - 1)) - 1
    shift = n - 3  # so that (s >> shift) exposes (b1,b2,b3) in the low bits

    for s in range(N):
        t = s >> shift  # low bits: ... b1 b2 b3
        # newbit = b1 & (b2 ^ b3)
        newbit = ((t >> 2) & 1) & (((t >> 1) & 1) ^ (t & 1))
        ns = ((s & mask) << 1) | newbit
        succ[s] = ns
        indeg[ns] += 1

    return succ, indeg


def S(n: int, mod: int = MOD) -> int:
    """Compute S(n) modulo mod."""
    if n < 3:
        raise ValueError("n must be >= 3")

    N = 1 << n
    succ, indeg = _build_successor_and_indegree(n)

    # Tree DP accumulators:
    # acc0[v] = product over processed children u of (dp0[u] + dp1[u])
    # acc1[v] = product over processed children u of dp0[u]
    # For leaves, both start at 1.
    acc0 = array("I", [1]) * N
    acc1 = array("I", [1]) * N

    # Identify cycle nodes with indegree-pruning (Kahn-style).
    # Nodes removed are NOT in cycles; remaining nodes are exactly the cycles.
    in_cycle = bytearray(b"\x01") * N
    q = deque()
    for i in range(N):
        if indeg[i] == 0:
            q.append(i)

    while q:
        u = q.popleft()
        if in_cycle[u] == 0:
            continue
        in_cycle[u] = 0  # pruned => not in a directed cycle

        p = succ[u]  # parent toward the cycle
        dp0 = acc0[u]
        dp1 = acc1[u]

        # Attach u as a child of p in the rooted in-tree.
        acc0[p] = (acc0[p] * ((dp0 + dp1) % mod)) % mod
        acc1[p] = (acc1[p] * dp0) % mod

        indeg[p] -= 1
        if indeg[p] == 0:
            q.append(p)

    # Now compute the independent-set count on each cycle, multiplying components.
    visited = bytearray(N)
    ans = 1

    for v in range(N):
        if in_cycle[v] and not visited[v]:
            # Enumerate the cycle in successor order.
            cycle = []
            u = v
            while not visited[u]:
                visited[u] = 1
                cycle.append(u)
                u = succ[u]

            k = len(cycle)

            # Weight of each cycle node after absorbing its attached in-tree:
            # - If node is NOT selected: weight0 = acc0[node]
            # - If node IS selected:     weight1 = acc1[node]
            w0 = [int(acc0[x]) % mod for x in cycle]
            w1 = [int(acc1[x]) % mod for x in cycle]

            # DP on a cycle with "no adjacent selected" constraint.
            # Case 1: first node not selected.
            prev0, prev1 = w0[0], 0
            for i in range(1, k):
                cur0 = ((prev0 + prev1) % mod) * w0[i] % mod
                cur1 = prev0 * w1[i] % mod
                prev0, prev1 = cur0, cur1
            case1 = (prev0 + prev1) % mod

            # Case 2: first node selected => last node must not be selected.
            prev0, prev1 = 0, w1[0]
            for i in range(1, k):
                cur0 = ((prev0 + prev1) % mod) * w0[i] % mod
                cur1 = prev0 * w1[i] % mod
                prev0, prev1 = cur0, cur1
            case2 = prev0  # force last node not selected

            ans = (ans * ((case1 + case2) % mod)) % mod

    return ans


def main() -> None:
    # Test values given in the problem statement:
    assert S(3) == 35
    assert S(4) == 2118

    print(S(20))


if __name__ == "__main__":
    main()
