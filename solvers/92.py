#!/usr/bin/env python
"""
@formalsw: theorem terminal_source
  (start cache : Nat) (entries : solvers.«92».store.Entries)
  (heap : FormalSW.PyAst.Mutable.Heap) (globals : FormalSW.PyAst.Mutable.Globals) :
  solvers.«92».store.Found heap cache entries →
    ProjectEulerSolutions.DigitSquareChains.Memo.Valid (solvers.«92».store.lookup entries) →
      solvers.«92».terminal.Verified start cache entries heap globals
"""


def next_term(n: int) -> int:
    """
    @formalsw: Py from project_euler_statements.p92 import next_term as reference_next_term
    @formalsw: Requires(n >= 0)
    @formalsw: Returns
    @formalsw: Equiv(reference_next_term)
    """
    total = 0
    while n:
        n, digit = divmod(n, 10)
        total += digit * digit
    return total


def terminal(start: int, known: dict[int, int]) -> int:
    """
    Share the endpoint of each digit-square chain with its preceding terms.
    """
    path = []
    positions = set()
    current = start
    while current not in known and current not in positions:
        positions.add(current)
        path.append(current)
        current = next_term(current)
    end = known.get(current, 0)
    for value in path:
        known[value] = end
    return end


def solve(limit_exclusive: int = 10_000_000) -> int:
    """
    Count digit-square chains reaching 89 by grouping equal digit-square sums.
    """
    if limit_exclusive <= 1:
        return 0
    counts = {(0, True): 1}
    for character in str(limit_exclusive - 1):
        ceiling = int(character)
        following = {}
        for (total, tight), ways in counts.items():
            for digit in range((ceiling if tight else 9) + 1):
                key = (total + digit * digit, tight and digit == ceiling)
                following[key] = following.get(key, 0) + ways
        counts = following
    known = {0: 0, 1: 1, 89: 89}
    return sum(ways for (total, tight), ways in counts.items() if terminal(total, known) == 89)


if __name__ == "__main__":
    known = {0: 0, 1: 1, 89: 89}
    assert terminal(44, known) == 1
    assert terminal(85, known) == 89
    assert solve(3) == 1
    print(solve())
