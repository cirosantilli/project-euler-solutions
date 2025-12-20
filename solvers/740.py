#!/usr/bin/env python3
"""
Project Euler 740: Secret Santa

We compute q(n): the probability that the *last* person ends up with at least one slip
with their own name, in the "two slips per person" variant described in the problem.

No external libraries are used (standard library only).
"""

from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Dict, Tuple


State = Tuple[int, int, int, int]  # (u1, u2, k, sp)


def q(n: int) -> float:
    """
    Return q(n) as a float.

    State compression:
      u1 = number of unprocessed (not-yet-drawn) non-last people with 1 slip remaining in the hat
      u2 = number of unprocessed non-last people with 2 slips remaining in the hat
      k  = number of slips of the last person remaining in the hat (0..2)
      sp = number of slips in the hat that belong to already-processed people

    At step t (0-based), there are m = (n-1 - t) unprocessed non-last people and
    T = 2n - 2t slips remaining in the hat.
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    if n == 2:
        # Person A cannot draw A; they draw B twice, leaving A slips for the last person -> failure certain.
        return 1.0

    # Initially: last person has 2 slips; all other (n-1) people are unprocessed with 2 slips each; no processed slips.
    dist: Dict[State, float] = {(0, n - 1, 2, 0): 1.0}

    for t in range(n - 1):  # process all non-last people
        m = (n - 1) - t
        T_total = 2 * n - 2 * t  # total slips remaining before this person's two draws

        newdist: DefaultDict[State, float] = defaultdict(float)

        for (u1, u2, k, sp), prob in dist.items():
            u0 = m - u1 - u2  # unprocessed non-last with 0 slips remaining

            # Pick the next person to draw (exchangeability lets us treat it as a uniformly random unprocessed person).
            for s, cnt in ((0, u0), (1, u1), (2, u2)):
                if cnt == 0:
                    continue
                p_actor = prob * (cnt / m)

                # Remove actor from unprocessed counts; their s slips remain in the hat but are excluded from their draws.
                uu1, uu2 = u1, u2
                if s == 1:
                    uu1 -= 1
                elif s == 2:
                    uu2 -= 1

                # Two sequential draws, each uniformly among slips not labelled with the actor's own name.
                # Draw 1:
                C1 = T_total - s
                invC1 = 1.0 / C1

                first_outcomes = []
                if k:
                    first_outcomes.append(
                        (k * invC1, uu1, uu2, k - 1, sp)
                    )  # drew last-person slip
                if uu1:
                    first_outcomes.append(
                        (uu1 * invC1, uu1 - 1, uu2, k, sp)
                    )  # drew from unprocessed(1)
                if uu2:
                    first_outcomes.append(
                        (2 * uu2 * invC1, uu1 + 1, uu2 - 1, k, sp)
                    )  # drew from unprocessed(2)
                if sp:
                    first_outcomes.append(
                        (sp * invC1, uu1, uu2, k, sp - 1)
                    )  # drew from processed pool

                # Draw 2 (after one slip removed):
                for p1, u1_1, u2_1, k_1, sp_1 in first_outcomes:
                    C2 = (T_total - 1) - s
                    invC2 = 1.0 / C2

                    # Enumerate second-draw outcomes
                    if k_1:
                        sp_new = sp_1 + s
                        newdist[(u1_1, u2_1, k_1 - 1, sp_new)] += (
                            p_actor * p1 * (k_1 * invC2)
                        )
                    if u1_1:
                        sp_new = sp_1 + s
                        newdist[(u1_1 - 1, u2_1, k_1, sp_new)] += (
                            p_actor * p1 * (u1_1 * invC2)
                        )
                    if u2_1:
                        sp_new = sp_1 + s
                        newdist[(u1_1 + 1, u2_1 - 1, k_1, sp_new)] += (
                            p_actor * p1 * (2 * u2_1 * invC2)
                        )
                    if sp_1:
                        sp_new = (sp_1 - 1) + s
                        newdist[(u1_1, u2_1, k_1, sp_new)] += (
                            p_actor * p1 * (sp_1 * invC2)
                        )

        dist = newdist

    # After processing n-1 people, only 2 slips remain. The process fails iff k>0.
    return sum(prob for (u1, u2, k, sp), prob in dist.items() if k > 0)


def _self_test() -> None:
    # Values given in the problem statement (rounded to 10 decimal places).
    assert f"{q(3):.10f}" == "0.3611111111"
    assert f"{q(5):.10f}" == "0.2476095994"


def main() -> None:
    _self_test()
    ans = q(100)
    print(f"{ans:.10f}")


if __name__ == "__main__":
    main()
