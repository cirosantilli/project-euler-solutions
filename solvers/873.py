#!/usr/bin/env python3
"""
Project Euler 873 - Words with Gaps

We count words containing exactly:
  - p copies of 'A'
  - q copies of 'B'
  - r copies of 'C'

Constraint: every 'A' is separated from every 'B' by at least two 'C's.

This program prints W(10^6, 10^7, 10^8) modulo 1_000_000_007.
It also asserts the two given test values from the problem statement.
"""

from __future__ import annotations

from array import array
from math import comb

MOD = 1_000_000_007


def w_exact(p: int, q: int, r: int) -> int:
    """
    Exact computation using a closed form (suitable for small parameters).
    """
    if p < 0 or q < 0 or r < 0:
        return 0
    if p == 0 and q == 0:
        return 1
    if p == 0 or q == 0:
        # No A-B interaction; just multinomial count.
        n = p + q + r
        # choose positions of C then positions of the remaining letter
        return comb(n, r) * comb(n - r, p)

    n = p + q
    tmax_transitions = 2 * min(p, q) - (1 if p == q else 0)
    tmax = min(tmax_transitions, r // 2)

    total = 0
    for t in range(0, tmax + 1):
        runs = t + 1

        # start with A
        a_runs = (runs + 1) // 2
        b_runs = runs // 2
        cnt = 0
        if a_runs <= p and b_runs <= q and a_runs > 0 and b_runs > 0:
            cnt += comb(p - 1, a_runs - 1) * comb(q - 1, b_runs - 1)

        # start with B
        a_runs2 = runs // 2
        b_runs2 = (runs + 1) // 2
        if a_runs2 <= p and b_runs2 <= q and a_runs2 > 0 and b_runs2 > 0:
            cnt += comb(p - 1, a_runs2 - 1) * comb(q - 1, b_runs2 - 1)

        if cnt == 0:
            continue

        # Distribute C's into (n+1) gaps, but each of the t transition gaps needs >=2 Cs.
        # Remaining: r' = r - 2t Cs distributed freely.
        total += cnt * comb((r - 2 * t) + n, n)

    return total


def _batch_inverses_consecutive(start: int, end: int, mod: int) -> array:
    """
    Compute modular inverses of all integers in the inclusive range [start, end],
    using one modular exponentiation (batch inversion) and O(end-start) multiplications.

    Precondition: 1 <= start <= end < mod.
    Returns an array invs where invs[i] == (start+i)^(-1) mod mod.
    """
    length = end - start + 1
    pref = array("I", [0]) * (length + 1)
    pref[0] = 1
    for i in range(length):
        pref[i + 1] = (pref[i] * (start + i)) % mod

    inv_total = pow(pref[length], mod - 2, mod)

    invs = array("I", [0]) * length
    for i in range(length - 1, -1, -1):
        x = start + i
        invs[i] = (inv_total * pref[i]) % mod
        inv_total = (inv_total * x) % mod

    return invs


def w_mod(p: int, q: int, r: int, mod: int = MOD) -> int:
    """
    Compute W(p,q,r) modulo mod.

    This uses:
      - counting AB-strings by number of transitions t (runs - 1)
      - stars-and-bars for distributing Cs into gaps with lower bounds
      - a fast recurrence for successive binomial values as t increases
      - batch inversion to avoid per-step pow() for modular division
    """
    if p < 0 or q < 0 or r < 0:
        return 0
    if p == 0 and q == 0:
        return 1
    if p == 0 or q == 0:
        # No interaction between A and B.
        n = p + q + r
        return (comb(n, r) * comb(n - r, p)) % mod

    # Symmetry: W(p,q,r) == W(q,p,r). Ensure p <= q to minimize precomputation.
    if p > q:
        p, q = q, p

    k = p + q  # total number of non-C letters
    # Maximum possible transitions in an AB-string with counts (p,q):
    tmax_transitions = 2 * p - (1 if p == q else 0)
    # Also need 2 Cs per transition.
    tmax = min(tmax_transitions, r // 2)
    if tmax <= 0:
        return 0

    # Precompute inverses 1..p+1 (needed for binomial-row recurrences).
    inv_small = array("I", [0]) * (p + 2)
    inv_small[1] = 1
    for i in range(2, p + 2):
        inv_small[i] = (mod - (mod // i) * inv_small[mod % i] % mod) % mod

    # Precompute C(p-1, x) for x=0..p-1  (array length p).
    choose_p = array("I", [0]) * p
    choose_p[0] = 1
    c = 1
    # C(n,k+1) = C(n,k) * (n-k)/(k+1)
    n1 = p - 1
    for x in range(0, p - 1):
        c = (c * (n1 - x)) % mod
        c = (c * inv_small[x + 1]) % mod
        choose_p[x + 1] = c

    # Precompute C(q-1, x) for x=0..p (array length p+1).
    choose_q = array("I", [0]) * (p + 1)
    choose_q[0] = 1
    c = 1
    n2 = q - 1
    for x in range(0, p):
        c = (c * (n2 - x)) % mod
        c = (c * inv_small[x + 1]) % mod
        choose_q[x + 1] = c

    # We need fast updates for:
    #   B_t = C((r - 2t) + k, k)
    #
    # Let N_t = (r - 2t) + k. Then N_{t} decreases by 2 each step.
    # Recurrence:
    #   C(N-2, k) = C(N, k) * (N-k)(N-k-1) / (N(N-1))
    #
    # We'll start with B_0 = C(r + k, k) and update forward to B_1..B_tmax.
    # Divisions are handled via modular inverses, precomputed for all needed N and N-1.

    N0 = r + k  # N at t=0
    # In the update for step t (from t-1 -> t), we divide by (N_{t-1})(N_{t-1}-1).
    # N_{t-1} ranges from N0 down to N0 - 2*(tmax-1).
    lowest_needed = (
        N0 - 2 * (tmax - 1) - 1
    )  # need inverses down to this (for N_{t-1}-1 at last step)
    inv_base = lowest_needed
    inv_end = N0
    inv_range = _batch_inverses_consecutive(inv_base, inv_end, mod)

    def inv_of(x: int) -> int:
        return inv_range[x - inv_base]

    # Compute B_0 = C(r+k, k) = prod_{i=1..k} (r+i)/i
    num = 1
    den = 1
    rr = r
    for i in range(1, k + 1):
        num = (num * (rr + i)) % mod
        den = (den * i) % mod
    B = (num * pow(den, mod - 2, mod)) % mod  # B_0

    ans = 0
    n_total = N0  # current N_{t-1} in update step
    n_gap = r  # current (N_{t-1} - k) == r - 2*(t-1)

    # Iterate t = 1..tmax:
    #   - update B to B_t using n_total, n_gap (from t-1)
    #   - add N_t * B_t
    for t in range(1, tmax + 1):
        # Update B from t-1 to t
        ratio = (n_gap % mod) * ((n_gap - 1) % mod) % mod
        ratio = ratio * inv_of(n_total) % mod
        ratio = ratio * inv_of(n_total - 1) % mod
        B = (B * ratio) % mod

        # Advance N components (now correspond to t)
        n_total -= 2
        n_gap -= 2

        # Count AB-strings with exactly t transitions (runs = t+1).
        # For runs = t+1:
        #   start with first letter: a_runs = ceil((t+1)/2), b_runs = floor((t+1)/2)
        #   start with second letter: swap
        a_runs = (t + 2) // 2
        b_runs = (t + 1) // 2
        count_t = 0
        if a_runs <= p and b_runs <= q:
            count_t = (choose_p[a_runs - 1] * choose_q[b_runs - 1]) % mod

        a_runs2 = (t + 1) // 2
        b_runs2 = (t + 2) // 2
        if a_runs2 <= p and b_runs2 <= q:
            count_t = (count_t + choose_p[a_runs2 - 1] * choose_q[b_runs2 - 1]) % mod

        ans = (ans + count_t * B) % mod

    return ans


def main() -> None:
    # Given test values (exact):
    assert w_exact(2, 2, 4) == 32
    assert w_exact(4, 4, 44) == 13908607644

    # Target computation:
    p = 10**6
    q = 10**7
    r = 10**8
    print(w_mod(p, q, r, MOD))


if __name__ == "__main__":
    main()
