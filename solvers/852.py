#!/usr/bin/env python3
"""Project Euler 852: Coins in a Box

Computes S(N) (expected final score with optimal play), rounded to 6 decimals.

No external libraries are used.
"""

from __future__ import annotations

import sys
from math import gcd


# A conservative truncation for the (infinite-horizon) per-coin optimal stopping problem.
# At MAX_FLIPS=180 the final answer for N=50 is stable to 6 decimals.
MAX_FLIPS = 180


def _stop_value(p: float) -> float:
    """Expected net gain if we stop now and guess optimally, given P(unfair)=p."""
    # Guess unfair if p >= 1/2, else guess fair.
    # Expected value if guess unfair: 70p - 50
    # Expected value if guess fair:   20 - 70p
    return 70.0 * p - 50.0 if p >= 0.5 else 20.0 - 70.0 * p


def _compute_g_for_fraction(
    a: int, b: int, pow3: list[float], inv2: list[float]
) -> float:
    """Compute G(p) for p=a/b, where G is optimal expected score for a single coin round.

    Uses backward DP on states (n,h): after n tosses with h heads.
    Terminal condition at n=MAX_FLIPS: forced stop.
    """
    if a <= 0 or a >= b:
        # Coin type is known with certainty.
        return 20.0

    p0 = a / b
    odds0 = p0 / (1.0 - p0)

    # next_row[h] represents the optimal value at state (n+1, h) for the current iteration.
    next_row = [0.0] * (MAX_FLIPS + 2)
    curr_row = [0.0] * (MAX_FLIPS + 2)

    # Initialize at n = MAX_FLIPS with forced stop.
    base = odds0 * inv2[MAX_FLIPS]
    for h in range(MAX_FLIPS + 1):
        odds = base * pow3[h]
        p = odds / (1.0 + odds)
        next_row[h] = _stop_value(p)

    # Backward induction for n = MAX_FLIPS-1 .. 0.
    for n in range(MAX_FLIPS - 1, -1, -1):
        base = odds0 * inv2[n]
        for h in range(n + 1):
            odds = base * pow3[h]
            p = odds / (1.0 + odds)

            stop_v = _stop_value(p)

            # P(head | current posterior) = 0.5 + 0.25 p
            qh = 0.5 + 0.25 * p
            # Continue: pay 1 for the toss, then transition.
            cont_v = -1.0 + qh * next_row[h + 1] + (1.0 - qh) * next_row[h]

            curr_row[h] = stop_v if stop_v >= cont_v else cont_v

        next_row, curr_row = curr_row, next_row

    return next_row[0]


def _precompute_g_cache(n_max: int) -> dict[tuple[int, int], float]:
    """Precompute G(p) for every prior p=u/(u+f) that can occur with u,f<=n_max."""
    # Precompute powers needed to evaluate posterior odds = odds0 * 3^h * 2^{-n}.
    pow3 = [1.0] * (MAX_FLIPS + 1)
    for i in range(1, MAX_FLIPS + 1):
        pow3[i] = pow3[i - 1] * 3.0

    inv2 = [1.0] * (MAX_FLIPS + 1)
    for i in range(1, MAX_FLIPS + 1):
        inv2[i] = inv2[i - 1] * 0.5

    need: set[tuple[int, int]] = set()
    for u in range(n_max + 1):
        for f in range(n_max + 1):
            total = u + f
            if total == 0:
                continue
            g = gcd(u, total)
            need.add((u // g, total // g))

    cache: dict[tuple[int, int], float] = {}
    for a, b in need:
        cache[(a, b)] = _compute_g_for_fraction(a, b, pow3, inv2)

    return cache


def solve(n: int = 50) -> float:
    """Return S(n): optimal expected final score for a game with n unfair and n fair coins."""
    if n < 0:
        raise ValueError("n must be non-negative")

    g_cache = _precompute_g_cache(n)

    # V[u][f] = expected optimal score from state with u unfair and f fair coins remaining.
    V = [[0.0] * (n + 1) for _ in range(n + 1)]

    # Fill by increasing total = u+f.
    for total in range(1, 2 * n + 1):
        u_min = max(0, total - n)
        u_max = min(n, total)
        for u in range(u_min, u_max + 1):
            f = total - u

            g = gcd(u, total)
            immediate = g_cache[(u // g, total // g)]

            exp_next = 0.0
            if u:
                exp_next += (u / total) * V[u - 1][f]
            if f:
                exp_next += (f / total) * V[u][f - 1]

            V[u][f] = immediate + exp_next

    return V[n][n]


def _self_test() -> None:
    # From the problem statement:
    # S(1) = 20.558591 (rounded to 6 digits after the decimal point)
    s1 = solve(1)
    assert round(s1, 6) == 20.558591


def main() -> None:
    _self_test()

    n = 50
    if len(sys.argv) >= 2:
        n = int(sys.argv[1])

    ans = solve(n)
    print(f"{ans:.6f}")


if __name__ == "__main__":
    main()
