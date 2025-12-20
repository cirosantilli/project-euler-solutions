#!/usr/bin/env python3
"""Project Euler 863: Different Dice

We want an optimal procedure (minimum expected rolls) to emulate a fair n-sided die
using only fair 5- and 6-sided dice, under the restriction that the *sequence of
dice rolled* is predetermined.

Model
-----
Maintain a uniform integer in [0, s-1] (s equally likely states).
Rolling an m-sided die expands this to [0, s*m-1]. If s*m >= n, we can map the
largest multiple of n states evenly onto the n outputs and "recycle" the rest.
Let r = (s*m) % n. Then, after one roll:
  * we stop with probability 1 - r/(s*m)
  * we continue in state-size r with probability r/(s*m)

Under this maximal-acceptance rule, the only relevant information is the current
state-size s in {0,1,...,n-1}; state 0 is terminal.

Optimality
----------
Let E[s] be the minimal expected remaining rolls from state-size s.
For s>0:
  E[s] = min_{m in {5,6}} ( 1 + (r/(s*m)) * E[r] ), where r = (s*m) % n
and E[0] = 0.

For each n we solve these Bellman optimality equations by Gauss-Seidel value
iteration, which converges very quickly for n <= 1000.

No external libraries are used.
"""

from __future__ import annotations


def R(n: int) -> float:
    """Return R(n): minimal expected number of rolls to emulate an n-sided die."""
    if n < 2:
        raise ValueError("n must be >= 2")

    # Precompute transitions for each choice of die.
    nxt5 = [0] * n
    nxt6 = [0] * n
    c5 = [0.0] * n
    c6 = [0.0] * n

    for s in range(1, n):
        t = 5 * s
        r = t % n
        nxt5[s] = r
        c5[s] = 0.0 if r == 0 else (r / t)

        t = 6 * s
        r = t % n
        nxt6[s] = r
        c6[s] = 0.0 if r == 0 else (r / t)

    # Gauss-Seidel value iteration.
    E = [0.0] * n
    eps = 1e-15

    for _ in range(10000):
        delta = 0.0
        # Descending sweep helps because many transitions with t < n go to larger s.
        for s in range(n - 1, 0, -1):
            v0 = 1.0 + c5[s] * E[nxt5[s]]
            v1 = 1.0 + c6[s] * E[nxt6[s]]
            new = v0 if v0 <= v1 else v1
            diff = new - E[s]
            if diff < 0:
                diff = -diff
            if diff > delta:
                delta = diff
            E[s] = new
        if delta < eps:
            break
    else:
        raise RuntimeError("value iteration failed to converge")

    return E[1]


def S(n: int) -> float:
    """Return S(n) = sum_{k=2..n} R(k)."""
    if n < 2:
        return 0.0
    total = 0.0
    for k in range(2, n + 1):
        total += R(k)
    return total


def _run_self_tests() -> None:
    # Test values from the problem statement (rounded to 6 decimal places).
    assert f"{R(8):.6f}" == "2.083333"
    assert f"{R(28):.6f}" == "2.142476"
    assert f"{S(30):.6f}" == "56.054622"


def main() -> None:
    _run_self_tests()
    ans = S(1000)
    print(f"{ans:.6f}")


if __name__ == "__main__":
    main()
