#!/usr/bin/env python3
"""
Project Euler 755: Not Zeckendorf

We define f(n) as the number of ways to write n as a sum of *distinct* Fibonacci
numbers from the sequence 1,2,3,5,8,13,... (i.e. F1=1, F2=2, Fi=Fi-1+Fi-2).

S(N) = sum_{k=0..N} f(k) equals the number of subsets of Fibonacci numbers whose
subset-sum is <= N.

This program computes S(10^13) without external libraries.
"""

from __future__ import annotations

from functools import lru_cache


def fib_upto(n: int) -> list[int]:
    """Return Fibonacci list with indexing: F[1]=1, F[2]=2, and containing values > n too."""
    if n < 0:
        raise ValueError("n must be non-negative")
    F = [0, 1, 2]
    while F[-1] <= n:
        F.append(F[-1] + F[-2])
    return F  # last element is > n


def S(n: int) -> int:
    """
    Count the number of subsets of Fibonacci numbers whose sum is <= n.
    Equivalently, S(n) = sum_{k=0..n} f(k).

    Uses recursion with memoization and a strong pruning bound:
    sum_{i=1..k} F_i = F_{k+2} - 2
    """
    if n < 0:
        return 0
    # Fibonacci numbers up to n
    F = fib_upto(n)
    k = len(F) - 2  # largest index with F[k] <= n

    # Ensure we can access F[k+2] for the sum formula
    while len(F) <= k + 2:
        F.append(F[-1] + F[-2])

    def sum_all(k_: int) -> int:
        """Sum of F1..Fk_."""
        if k_ <= 0:
            return 0
        return F[k_ + 2] - 2

    @lru_cache(maxsize=None)
    def count(k_: int, cap: int) -> int:
        """Number of subsets of {F1..Fk_} with subset-sum <= cap."""
        if cap < 0:
            return 0
        if k_ == 0:
            return 1  # empty subset only
        # If even the maximal possible sum fits, all subsets are valid.
        if cap >= sum_all(k_):
            return 1 << k_
        # If the current Fibonacci is too large, we cannot take it.
        if cap < F[k_]:
            return count(k_ - 1, cap)
        # Otherwise: exclude or include F[k_]
        return count(k_ - 1, cap) + count(k_ - 1, cap - F[k_])

    return count(k, n)


def main() -> None:
    # Test values from the problem statement
    assert S(100) == 415
    assert S(10_000) == 312_807

    target = 10**13
    print(S(target))


if __name__ == "__main__":
    main()
