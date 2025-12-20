#!/usr/bin/env python3
"""
Project Euler 815: Group by Value

A deck has 4n cards: 4 identical copies of each value 1..n.
Cards are dealt in random order. Each value has (at most) one pile:
- If a dealt card's value has a pile, it is added to that pile.
- Otherwise, it starts a new pile.
- When a pile reaches 4 cards, it is removed.

Let E(n) be the expected maximum number of non-empty piles observed during the process.
Compute E(60) and print it rounded to 8 digits after the decimal point.

No external libraries are used.
"""

from __future__ import annotations

from array import array
from math import comb, fsum
import sys


class Model:
    __slots__ = (
        "n",
        "total",
        "max_rem",
        "buckets",  # buckets[remaining_cards][active_piles] -> list[state_index]
        "inv",
        "next0",
        "num0",
        "next1",
        "num1",
        "next2",
        "num2",
        "next3",
        "num3",
        "start",
    )

    def __init__(self, n: int) -> None:
        self.n = n
        self.total = comb(n + 4, 4)  # number of (x0,x1,x2,x3) with sum <= n
        self.max_rem = 4 * n

        # Buckets to iterate states in increasing "remaining cards" order,
        # and (for threshold k) only over active piles < k.
        self.buckets = [[[] for _ in range(n + 1)] for __ in range(self.max_rem + 1)]

        # Compact arrays for state data.
        self.inv = array("d", [0.0]) * self.total

        self.next0 = array("I", [0]) * self.total
        self.next1 = array("I", [0]) * self.total
        self.next2 = array("I", [0]) * self.total
        self.next3 = array("I", [0]) * self.total

        self.num0 = array("H", [0]) * self.total
        self.num1 = array("H", [0]) * self.total
        self.num2 = array("H", [0]) * self.total
        self.num3 = array("H", [0]) * self.total

        # Indexing scheme over all states:
        # state is (x0,x1,x2,x3) where
        # x0 = #values with 0 copies seen (4 remaining in deck)
        # x1 = #values with 1 copy seen (3 remaining)  -> pile size 1
        # x2 = #values with 2 copies seen (2 remaining) -> pile size 2
        # x3 = #values with 3 copies seen (1 remaining) -> pile size 3
        # x4 = n - (x0+x1+x2+x3) values completed (0 remaining), implicit
        #
        # We order by s = x0+x1+x2+x3 increasing, then lexicographically x0,x1,x2
        # (x3 is determined).

        prefix_s = [0] * (n + 2)
        for s in range(n + 1):
            prefix_s[s + 1] = prefix_s[s] + comb(s + 3, 3)

        x0_pref = []
        for s in range(n + 1):
            arr = [0] * (s + 2)
            running = 0
            for x0 in range(s + 1):
                arr[x0] = running
                t = s - x0
                running += comb(t + 2, 2)  # solutions to x1+x2+x3=t
            arr[s + 1] = running
            x0_pref.append(arr)

        def idx(x0: int, x1: int, x2: int, x3: int) -> int:
            s = x0 + x1 + x2 + x3
            t = s - x0
            base = prefix_s[s] + x0_pref[s][x0]
            # offset within fixed (s,x0) by x1, then x2
            base += x1 * (t + 1) - (x1 * (x1 - 1)) // 2
            base += x2
            return base

        # Enumerate all states in the same order as idx().
        i = 0
        for s in range(n + 1):
            for x0 in range(s + 1):
                t = s - x0
                for x1 in range(t + 1):
                    u = t - x1
                    for x2 in range(u + 1):
                        x3 = u - x2

                        remaining = 4 * x0 + 3 * x1 + 2 * x2 + x3
                        active = x1 + x2 + x3

                        self.inv[i] = 0.0 if remaining == 0 else 1.0 / remaining
                        self.buckets[remaining][active].append(i)

                        # Transition numerators are the number of remaining cards of that category:
                        # - from x0: 4*x0
                        # - from x1: 3*x1
                        # - from x2: 2*x2
                        # - from x3: 1*x3
                        # Each draw reduces remaining by exactly 1.
                        if x0:
                            self.next0[i] = idx(x0 - 1, x1 + 1, x2, x3)
                            self.num0[i] = 4 * x0
                        if x1:
                            self.next1[i] = idx(x0, x1 - 1, x2 + 1, x3)
                            self.num1[i] = 3 * x1
                        if x2:
                            self.next2[i] = idx(x0, x1, x2 - 1, x3 + 1)
                            self.num2[i] = 2 * x2
                        if x3:
                            self.next3[i] = idx(x0, x1, x2, x3 - 1)
                            self.num3[i] = x3

                        i += 1

        if i != self.total:
            raise RuntimeError("State enumeration mismatch")

        # Start state: all values unseen => (n,0,0,0)
        self.start = idx(n, 0, 0, 0)
        # Terminal state is (0,0,0,0) which is index 0 in our order.


def probability_max_less_than_k(model: Model, k: int) -> float:
    """
    Return P(maximum active piles < k) starting from the initial state.

    We compute this by dynamic programming on the acyclic state graph:
    every transition reduces the total number of remaining cards by 1.
    """
    n = model.n
    total = model.total
    max_rem = model.max_rem
    buckets = model.buckets
    inv = model.inv

    next0, num0 = model.next0, model.num0
    next1, num1 = model.next1, model.num1
    next2, num2 = model.next2, model.num2
    next3, num3 = model.next3, model.num3

    dp = [0.0] * total
    dp[0] = 1.0  # terminal state

    max_active = k - 1
    if max_active > n:
        max_active = n

    # Process states by increasing remaining cards so that dp[next_state] is already known.
    for rem in range(1, max_rem + 1):
        # Only states with active piles < k are valid; others have probability 0 (left as default).
        for active in range(max_active + 1):
            for s in buckets[rem][active]:
                dp[s] = (
                    num0[s] * dp[next0[s]]
                    + num1[s] * dp[next1[s]]
                    + num2[s] * dp[next2[s]]
                    + num3[s] * dp[next3[s]]
                ) * inv[s]

    return dp[model.start]


def expected_max_piles(n: int) -> float:
    """
    Compute E(n) as sum_{k=1..n} P(max >= k) = sum_{k=1..n} (1 - P(max < k)).
    """
    model = Model(n)
    probs = []
    for k in range(1, n + 1):
        probs.append(probability_max_less_than_k(model, k))
    return fsum(1.0 - p for p in probs)


def main() -> None:
    # Test value from the problem statement:
    # E(2) = 1.97142857 (rounded to 8 decimal places)
    e2 = expected_max_piles(2)
    assert f"{e2:.8f}" == "1.97142857"

    n = 60
    ans = expected_max_piles(n)
    print(f"{ans:.8f}")


if __name__ == "__main__":
    main()
