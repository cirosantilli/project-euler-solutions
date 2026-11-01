#!/usr/bin/env python
"""
@formalsw: theorem rational_source (numerator denominator : Int)
  (heap : FormalSW.PyAst.Mutable.Heap) (globals : FormalSW.PyAst.Mutable.Globals) :
  denominator ≠ 0 → solvers.«93».rational.Verified numerator denominator heap globals
@formalsw: theorem pair_results_source (a b : ProjectEulerSolutions.RationalPair.Pair)
  (left right : Nat) (heap : FormalSW.PyAst.Mutable.Heap) (globals : FormalSW.PyAst.Mutable.Globals) :
  0 < a.2 → 0 < b.2 →
    heap[left]? = some (ProjectEulerSolutions.RationalPair.Heap.object a) →
      heap[right]? = some (ProjectEulerSolutions.RationalPair.Heap.object b) →
        solvers.«93».pair_results.Verified a b left right heap globals
@formalsw: theorem recursive_base_source (heap : FormalSW.PyAst.Mutable.Heap)
  (globals : FormalSW.PyAst.Mutable.Globals) : solvers.«93».all_results.BaseCasesVerified heap globals
@formalsw: theorem recursive_source (heap : FormalSW.PyAst.Mutable.Heap)
  (globals : FormalSW.PyAst.Mutable.Globals) : solvers.«93».all_results.RecursiveCallsVerified heap globals
"""
from __future__ import annotations

from itertools import combinations
from math import gcd


def rational(numerator: int, denominator: int) -> tuple[int, int]:
    """
    Canonicalize a rational with a nonzero denominator for tuple hashing.

    @formalsw: Requires(denominator != 0)
    @formalsw: Returns
    """
    if denominator < 0:
        numerator = -numerator
        denominator = -denominator
    divisor = gcd(numerator, denominator)
    return numerator // divisor, denominator // divisor


def pair_results(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    numerator_a, denominator_a = a
    numerator_b, denominator_b = b
    denominator = denominator_a * denominator_b
    results = [
        rational(numerator_a * denominator_b + numerator_b * denominator_a, denominator),
        rational(numerator_a * denominator_b - numerator_b * denominator_a, denominator),
        rational(numerator_b * denominator_a - numerator_a * denominator_b, denominator),
        rational(numerator_a * numerator_b, denominator),
    ]
    if numerator_b:
        results.append(rational(numerator_a * denominator_b, denominator_a * numerator_b))
    if numerator_a:
        results.append(rational(numerator_b * denominator_a, denominator_b * numerator_a))
    return results


def all_results(digits: tuple[int, ...]) -> set[tuple[int, int]]:
    """
    Reduce every pair of expression fragments, caching canonical multisets.
    """
    start = tuple(sorted((digit, 1) for digit in digits))
    memo = {}

    def rec(state: tuple[tuple[int, int], ...]) -> set[tuple[int, int]]:
        if state in memo:
            return memo[state]
        if len(state) == 1:
            memo[state] = {state[0]}
            return memo[state]

        results = set()
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                rest = [state[k] for k in range(len(state)) if k != i and k != j]
                for value in pair_results(state[i], state[j]):
                    results.update(rec(tuple(sorted(rest + [value]))))
        memo[state] = results
        return results

    return rec(start)


def consecutive_length(digits: tuple[int, ...]) -> tuple[int, set[int]]:
    integers = set()
    for numerator, denominator in all_results(digits):
        if denominator == 1 and numerator > 0:
            integers.add(numerator)
    n = 1
    while n in integers:
        n += 1
    return n - 1, integers


def solve(max_digit: int = 9) -> tuple[int, ...]:
    best_length = -1
    best_digits = ()
    for digits in combinations(range(max_digit + 1), 4):
        length, _ = consecutive_length(digits)
        if length > best_length:
            best_length = length
            best_digits = digits
    return best_digits


if __name__ == "__main__":
    length, integers = consecutive_length((1, 2, 3, 4))
    assert length == 28
    assert len(integers) == 31
    assert max(integers) == 36
    assert consecutive_length((1, 2, 5, 8))[0] == 51
    print("".join(map(str, solve())))
