#!/usr/bin/env python3
"""Project Euler 938: Exhausting a Colour

We need P(R,B): the probability that the game ends with only black cards
remaining, starting from R red and B black cards.

No external libraries are used.
"""

import math
import sys
from typing import List, Optional


def _log_add_exp(log_x: Optional[float], log_y: float) -> float:
    """Return log(exp(log_x) + exp(log_y)) given logs, stably.

    log_x may be None to indicate an empty sum.
    """
    if log_x is None:
        return log_y
    # Ensure log_x >= log_y
    if log_y > log_x:
        log_x, log_y = log_y, log_x
    # log_x + log(1 + exp(log_y-log_x))
    return log_x + math.log1p(math.exp(log_y - log_x))


def probability_black(R: int, B: int) -> float:
    """Compute P(R,B) as a float.

    Key observations:
    - Red cards are only removed in pairs, so if R is odd then reaching R=0
      is impossible => probability is 0.
    - Conditioning on the non-(BB) cases removes the self-loop and yields a
      recurrence that can be transformed into a Pascal-style sum.

    The final evaluation is done using lgamma + log-sum-exp for numerical
    stability and speed.
    """
    if B <= 0:
        return 0.0
    if R <= 0:
        return 1.0
    if R & 1:
        return 0.0

    # Work with a = R/2 (number of red pairs), b = B
    a = R // 2
    b = B
    if a == 0:
        return 1.0

    # Derived closed-form:
    #   P(2a,b) = u(a,b) / C(a,b)
    # where
    #   u(a,b) = sum_{k=1..b} g(k) * binom(a+b-k-1, a-1)
    #   g(k)   = binom(2k,k) / 4^k
    #   C(a,b) = Gamma(a+b+1/2) / (Gamma(a+1/2) * Gamma(b+1))
    # We compute logs and sum in log-space.

    log4 = math.log(4.0)
    m = a - 1

    log_u: Optional[float] = None
    for k in range(1, b + 1):
        # log(g(k)) = log(binomial(2k,k)) - k*log(4)
        log_g = math.lgamma(2 * k + 1) - 2.0 * math.lgamma(k + 1) - k * log4

        # binom(n,m) with n = a+b-k-1 and m = a-1
        n = a + b - k - 1
        log_binom = math.lgamma(n + 1) - math.lgamma(m + 1) - math.lgamma(n - m + 1)

        log_u = _log_add_exp(log_u, log_g + log_binom)

    # log(C(a,b))
    log_C = math.lgamma(a + b + 0.5) - math.lgamma(a + 0.5) - math.lgamma(b + 1)

    return math.exp(log_u - log_C)


def _self_test() -> None:
    # Test values from the problem statement.
    assert abs(probability_black(2, 2) - 0.4666666667) < 1e-10
    assert abs(probability_black(10, 9) - 0.4118903397) < 1e-10
    assert abs(probability_black(34, 25) - 0.3665688069) < 1e-10

    # Parity sanity: odd R can never be fully removed.
    assert probability_black(1, 1) == 0.0
    assert probability_black(3, 100) == 0.0


def main(argv: List[str]) -> None:
    _self_test()

    if len(argv) == 3:
        R = int(argv[1])
        B = int(argv[2])
    else:
        R = 24690
        B = 12345

    ans = probability_black(R, B)
    # Print with exactly 10 digits after the decimal point.
    print("{:.10f}".format(ans))


if __name__ == "__main__":
    main(sys.argv)
