#!/usr/bin/env python3
"""
Project Euler 874: Maximal Prime Score

Let p(t) be the (t+1)-th prime (p(0)=2, p(1)=3, ...).
For given k, n:
- choose a list [a1..an] with 0 <= ai < k
- such that sum(ai) is a multiple of k
- maximize sum(p(ai))

This program computes M(7000, p(7000)) and prints it.
No external libraries are used.
"""

from __future__ import annotations

import math


def first_n_primes(n: int) -> list[int]:
    """Return the first n primes in increasing order (n >= 0)."""
    if n <= 0:
        return []
    if n < 6:
        # Enough to cover small n exactly.
        limit = 15
    else:
        # Rosser–Schoenfeld style upper bound for nth prime:
        # p_n < n (log n + log log n) for n >= 6
        nn = float(n)
        limit = int(nn * (math.log(nn) + math.log(math.log(nn))) + 10)

    while True:
        sieve = bytearray(b"\x01") * (limit + 1)
        if limit >= 0:
            sieve[0:2] = b"\x00\x00"
        root = int(limit**0.5)
        for i in range(2, root + 1):
            if sieve[i]:
                step = i
                start = i * i
                sieve[start : limit + 1 : step] = b"\x00" * (
                    ((limit - start) // step) + 1
                )

        primes: list[int] = [i for i in range(2, limit + 1) if sieve[i]]
        if len(primes) >= n:
            return primes[:n]

        # If our bound was too small, increase and try again.
        limit *= 2


def min_prime_loss_for_reduction(primes_first_k: list[int], reduction: int) -> int:
    """
    We start from all a_i = k-1, so each element contributes prime_top = p(k-1).
    If we reduce one element by d (i.e. set it to k-1-d), the score decreases by:
        loss[d] = p(k-1) - p(k-1-d)

    Given a required total reduction 'reduction' (0 <= reduction < k),
    find the minimal total loss among all multisets of reductions that sum to it.
    Unbounded knapsack in O(reduction^2).
    """
    if reduction == 0:
        return 0

    m = len(primes_first_k) - 1  # m = k-1
    top = primes_first_k[m]

    # loss[d] for 1..reduction
    loss = [0] * (reduction + 1)
    for d in range(1, reduction + 1):
        loss[d] = top - primes_first_k[m - d]

    INF = 10**30
    dp = [INF] * (reduction + 1)
    dp[0] = 0

    # Unbounded knapsack (min-cost exact sum)
    for d in range(1, reduction + 1):
        c = loss[d]
        for s in range(d, reduction + 1):
            cand = dp[s - d] + c
            if cand < dp[s]:
                dp[s] = cand

    return dp[reduction]


def maximal_prime_score(k: int, n: int, primes_first_k: list[int] | None = None) -> int:
    """
    Compute M(k, n).

    Strategy:
    - The best unconstrained choice is a_i = k-1 for all i.
    - Let D = sum((k-1) - a_i) be the total reduction from that all-(k-1) baseline.
      Then sum(a_i) = n*(k-1) - D.
    - We need sum(a_i) ≡ 0 (mod k). Since (k-1) ≡ -1 (mod k),
      this forces D ≡ -n (mod k). The smallest nonnegative such D is r = (-n) mod k.
    - Any extra reduction by multiples of k only decreases the score, so D = r is optimal.
    - For that fixed reduction amount r, minimize the prime-score loss via knapsack.

    Returns the maximal score as an integer.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if n < 0:
        raise ValueError("n must be nonnegative")
    if k == 1:
        # Only possible value is 0; p(0)=2, sum is always multiple of 1
        return n * 2

    if primes_first_k is None:
        primes_first_k = first_n_primes(k)
    if len(primes_first_k) != k:
        raise ValueError("primes_first_k must contain exactly the first k primes")

    top = primes_first_k[k - 1]  # p(k-1)
    r = (-n) % k
    loss = min_prime_loss_for_reduction(primes_first_k, r)
    return n * top - loss


def _self_test() -> None:
    # Test value given in the problem statement
    assert maximal_prime_score(2, 5) == 14


def main() -> None:
    _self_test()

    k = 7000

    # Need p(7000), i.e. the 7001st prime, and also the first 7000 primes.
    primes_7001 = first_n_primes(k + 1)
    primes_first_k = primes_7001[:k]
    n = primes_7001[k]  # p(7000)

    ans = maximal_prime_score(k, n, primes_first_k)
    print(ans)


if __name__ == "__main__":
    main()
