#!/usr/bin/env python3
"""
Project Euler 762 — Amoebas in a 2D grid

We count distinct final arrangements after N divisions, modulo 1_000_000_000
(last nine digits).
"""

MOD = 1_000_000_000


def _popcount4(mask: int) -> int:
    # mask in [0, 15]
    return (mask & 1) + ((mask >> 1) & 1) + ((mask >> 2) & 1) + ((mask >> 3) & 1)


def _expand(prev):
    """
    Linear map A on a 4-vector:
      t[y] = prev[y] + prev[y-1 mod 4]
    """
    a0, a1, a2, a3 = prev
    return (a0 + a3, a1 + a0, a2 + a1, a3 + a2)


def _build_states():
    """
    States are shot-vectors s_x (how many times each cell in column x divides),
    and we only ever need sum(s_x) <= 3 (see README).
    """
    states = []
    idx = {}
    for a0 in range(4):
        for a1 in range(4 - a0):
            for a2 in range(4 - a0 - a1):
                for a3 in range(4 - a0 - a1 - a2):
                    t = (a0, a1, a2, a3)
                    idx[t] = len(states)
                    states.append(t)
    return states, idx


def _build_transitions(states, idx):
    """
    For a previous shot-vector s, we compute t = A(s).
    The final occupancy in the next column is a 4-bit mask p (0/1 per row),
    and the next shot-vector is s' = t - p (componentwise).
    Constraints:
      - p is 0/1 per component
      - s' is a valid state (nonnegative, each <= 3, sum <= 3)
    """
    S = len(states)
    terminal = idx[(0, 0, 0, 0)]
    pop = [_popcount4(m) for m in range(16)]

    to_nonterm = [[] for _ in range(S)]  # (v, weight)
    to_term_w = [[] for _ in range(S)]  # weight only

    for u, s in enumerate(states):
        if u == terminal:
            continue
        t = _expand(s)
        for mask in range(16):
            # s_next = t - bits(mask)
            b0 = mask & 1
            b1 = (mask >> 1) & 1
            b2 = (mask >> 2) & 1
            b3 = (mask >> 3) & 1
            n0 = t[0] - b0
            n1 = t[1] - b1
            n2 = t[2] - b2
            n3 = t[3] - b3
            if n0 < 0 or n1 < 0 or n2 < 0 or n3 < 0:
                continue
            if n0 > 3 or n1 > 3 or n2 > 3 or n3 > 3:
                continue
            if n0 + n1 + n2 + n3 > 3:
                continue
            v = idx.get((n0, n1, n2, n3))
            if v is None:
                continue
            w = pop[mask]
            if v == terminal:
                to_term_w[u].append(w)
            else:
                to_nonterm[u].append((v, w))

    return terminal, to_nonterm, to_term_w


def compute_all(max_n: int, mod: int = MOD):
    """
    Returns [C(0), C(1), ..., C(max_n)] modulo mod.
    """
    if max_n < 0:
        return []
    states, idx = _build_states()
    terminal, to_nonterm, to_term_w = _build_transitions(states, idx)

    # State order matters only for 0-weight transitions: they go from sum=1 to sum=2.
    order = sorted(range(len(states)), key=lambda i: (sum(states[i]), states[i]))

    # For N>0, (0,0) must divide at least once, so column 0 is empty and s_0 = (1,0,0,0).
    start = idx[(1, 0, 0, 0)]

    # Total amoebas after N divisions is N+1.
    mmax = max_n + 1

    # end[m] counts configurations that terminate (reach terminal shot-vector)
    # with exactly m amoebas in the final arrangement (excluding the N=0 special case).
    end = [0] * (mmax + 1)

    # Weighted automaton DP over "amoeba count so far".
    # Max weight per column is 4, so we only need 5 layers (ring buffer).
    layers = [[0] * len(states) for _ in range(5)]
    layers[0][start] = 1

    for m in range(mmax + 1):
        cur = layers[0]
        for u in order:
            val = cur[u]
            if not val:
                continue

            # Transitions that terminate.
            for w in to_term_w[u]:
                nm = m + w
                if nm <= mmax:
                    end[nm] = (end[nm] + val) % mod

            # Transitions to nonterminal states.
            for v, w in to_nonterm[u]:
                nm = m + w
                if nm > mmax:
                    continue
                if w == 0:
                    # Same m, but these only go from sum=1 to sum=2; order guarantees correctness.
                    cur[v] = (cur[v] + val) % mod
                else:
                    layers[w][v] = (layers[w][v] + val) % mod

        # Advance the ring buffer to m+1.
        layers.pop(0)
        layers.append([0] * len(states))

    C = [0] * (max_n + 1)
    C[0] = 1  # N=0: the original single amoeba at (0,0)
    for n in range(1, max_n + 1):
        C[n] = end[n + 1] % mod
    return C


def solve():
    target_n = 100_000
    C = compute_all(target_n)

    # Tests from the problem statement.
    assert C[2] == 2
    assert C[10] == 1301
    assert C[20] == 5895236
    assert C[100] == 125923036  # last nine digits

    print(f"{C[target_n]:09d}")  # last nine digits, zero-padded


if __name__ == "__main__":
    solve()
