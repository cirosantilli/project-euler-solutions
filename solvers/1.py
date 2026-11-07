#!/usr/bin/env python

# Nagini
from nagini_contracts.contracts import Ensures, Old, Pure, Requires
from nagini_contracts.obligations import MustTerminate
from project_euler_statements.p1 import naive
from p1_nagini import equivalence


# Nagini
@Pure
def sum_of_multiples(m: int, n: int) -> int:
    # Nagini
    Requires(m > 0)
    Requires(n >= 0)

    max_range = n // m
    return m * max_range * (max_range + 1) // 2


def solve(n: int) -> int:
    """
    @formalsw: Py from project_euler_statements.p1 import naive
    @formalsw: Requires(n > 1)
    @formalsw: Equiv(naive)
    @formalsw: BigO(1)
    """
    # Nagini
    Requires(n > 1)
    Requires(MustTerminate(1))
    Ensures(int, lambda result: equivalence(Old(n)) and result == naive(Old(n)))

    n -= 1
    return sum_of_multiples(3, n) + sum_of_multiples(5, n) - sum_of_multiples(15, n)


if __name__ == "__main__":
    assert solve(10) == 23
    print(solve(1000))
