#!/usr/bin/env python3
"""Project Euler 887: Bounded Binary Search.

We search for a secret x in {1..N} by repeatedly choosing y and asking:
    "Is the secret number greater than y?"

Let Q(N, d) be the least possible *worst-case* number of questions of a
strategy that can find any secret x in {1..N} while also satisfying:
    questions_used_to_find_x <= x + d

Task: compute
    sum_{d=0..7} sum_{N=1..7^10} Q(N, d)

Running this file prints the required value.
"""

from functools import lru_cache


@lru_cache(maxsize=None)
def max_n(q: int, d: int) -> int:
    """Return the maximum N solvable with <= q questions and slack d.

    Slack d means each secret x must be found within x + d questions.
    """
    if d < 0:
        # For x=1 we would need <= 1+d questions. If d<0, that's <=0,
        # so only N=1 is feasible.
        return 1
    if q == 0:
        return 1
    if d >= q - 1:
        # Even x=1 may take q questions, so the only limit is that q
        # yes/no answers distinguish at most 2^q secrets.
        return 1 << q

    # After the first question, the "no" branch contains the smallest values.
    # Those values have one fewer remaining question available, so its slack is d-1.
    left = max_n(q - 1, d - 1)

    # The "yes" branch is shifted upward by `left` values; this increases
    # its effective slack by `left` as well.
    return left + max_n(q - 1, left + d - 1)


def Q(N: int, d: int) -> int:
    """Minimum worst-case number of questions needed for (N,d)."""
    if N <= 1:
        return 0
    if d == 0:
        # Given in the problem statement.
        return N - 1

    q = 0
    while max_n(q, d) < N:
        q += 1
    return q


def sum_Q_upto(limit: int, d: int) -> int:
    """Compute sum_{N=1..limit} Q(N,d)."""
    if limit <= 0:
        return 0
    if d == 0:
        # sum_{N=1..limit} (N-1)
        return limit * (limit - 1) // 2

    total = 0
    prev_cap = 0
    q = 0
    while prev_cap < limit:
        cap = max_n(q, d)
        if cap > limit:
            cap = limit
        total += q * (cap - prev_cap)
        prev_cap = cap
        q += 1
    return total


def solve() -> int:
    limit = 7**10
    return sum(sum_Q_upto(limit, d) for d in range(8))


def _self_test() -> None:
    # Test values given in the problem statement:
    assert Q(7, 1) == 3
    assert Q(777, 2) == 10

    # Also given: Q(N,0) = N-1
    for n in range(1, 50):
        assert Q(n, 0) == n - 1

    # And a consistency check implied by Q(N,0)=N-1: max_n(q,0) == q+1
    for q in range(0, 40):
        assert max_n(q, 0) == q + 1


def main() -> None:
    _self_test()
    print(solve())


if __name__ == "__main__":
    main()
