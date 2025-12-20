#!/usr/bin/env python3
"""Project Euler 871: Drifting Subsets

Let f be a function from a finite set S to itself.
A subset A is "drifting" if |A ∪ f(A)| = 2|A|.

For n ≥ 1, define f_n on {0,1,...,n-1} by:
    f_n(x) = x^3 + x + 1 (mod n)

This program computes:
    sum_{i=1..100} D(f_{100000 + i})
where D(f) is the maximum size of a drifting subset for f.

No external libraries are used.
"""


def drifting_subset_size(n: int) -> int:
    """Compute D(f_n) for f_n(x) = x^3 + x + 1 (mod n)."""
    # Build the functional graph: out-edge x -> f[x], and indegrees.
    f = [0] * n
    indeg = [0] * n
    for x in range(n):
        y = (x * x * x + x + 1) % n
        f[x] = y
        indeg[y] += 1

    # Each component is a directed cycle with trees feeding into it.
    # We compute maximum matching sizes for all tree nodes by trimming indegree-0 nodes.
    # For each node v we maintain:
    #   sum_a[v] = sum of dp0(child) over processed predecessors (tree children)
    #   best[v]  = 1 if matching v with one child can improve dp0(v), else 0
    # Here dp0(v) = best matching in v-subtree when v is NOT matched to its parent.
    #      dp1(v) = best matching in v-subtree when v IS matched to its parent.
    # For rooted trees in this functional graph, dp0(v) - dp1(v) is always 0 or 1.
    sum_a = [0] * n
    best = bytearray(n)  # stores 0 or 1

    q = [i for i, d in enumerate(indeg) if d == 0]
    pop = q.pop
    append = q.append

    while q:
        u = pop()

        dp0_u = sum_a[u] + best[u]  # dp1_u == sum_a[u]
        v = f[u]

        # Accumulate u's contribution into its successor v.
        sum_a[v] += dp0_u

        # Gain of matching u to v equals 1 + dp1_u - dp0_u = 1 - best[u].
        # Since best[u] is 0/1, gain is 1 if best[u]==0 else 0.
        if best[u] == 0:
            best[v] = 1

        indeg[v] -= 1
        if indeg[v] == 0:
            append(v)

    # Remaining nodes (indeg>0) are exactly the cycle nodes.
    visited = bytearray(n)
    total = 0

    for start in range(n):
        if indeg[start] == 0 or visited[start]:
            continue

        # Traverse this cycle.
        cycle = []
        x = start
        while not visited[x]:
            visited[x] = 1
            cycle.append(x)
            x = f[x]

        k = len(cycle)
        if k == 1:
            # Self-loop: that edge cannot be used (would violate A ∩ f(A) = ∅).
            v = cycle[0]
            total += sum_a[v] + best[v]
            continue

        # For each cycle vertex v:
        #   occ(v)  = sum_a[v]                (best tree matching when v is occupied by cycle)
        #   delta(v)= best[v] in {0,1}        (extra edge if v matches a tree child)
        #   free(v) = occ(v) + delta(v)
        deltas = [0] * k
        base = 0
        for i, v in enumerate(cycle):
            d = best[v]
            deltas[i] = d
            base += sum_a[v] + d

        # Cycle edge i is (cycle[i] -> cycle[(i+1)%k]).
        # If we use that edge, we gain +1 but we must forgo deltas at both endpoints.
        # So its net weight over the base is: 1 - delta[i] - delta[i+1].

        # Case A: do not use the edge (k-1 -> 0). Solve a path on edges 0..k-2.
        dp2 = 0
        dp1 = 0
        for i in range(k - 1):
            w = 1 - deltas[i] - deltas[i + 1]
            take = dp2 + w
            dp = take if take > dp1 else dp1
            dp2, dp1 = dp1, dp
        gain_a = dp1

        # Case B: use the edge (k-1 -> 0); then edges adjacent to its endpoints are forbidden.
        w_last = 1 - deltas[k - 1] - deltas[0]
        if k <= 2:
            gain_b = w_last
        else:
            dp2 = 0
            dp1 = 0
            for i in range(1, k - 2):
                w = 1 - deltas[i] - deltas[i + 1]
                take = dp2 + w
                dp = take if take > dp1 else dp1
                dp2, dp1 = dp1, dp
            gain_b = w_last + dp1

        best_cycle_gain = gain_a
        if gain_b > best_cycle_gain:
            best_cycle_gain = gain_b
        if best_cycle_gain < 0:
            best_cycle_gain = 0

        total += base + best_cycle_gain

    return total


def solve() -> int:
    return sum(drifting_subset_size(100_000 + i) for i in range(1, 101))


def _self_test() -> None:
    # Given in the problem statement.
    assert drifting_subset_size(5) == 1
    assert drifting_subset_size(10) == 3


if __name__ == "__main__":
    _self_test()
    print(solve())
