#!/usr/bin/env python3
"""Project Euler 856: Waiting for a Pair

Compute the expected number of cards drawn from a standard 52-card deck until two
consecutive cards share the same rank, or the deck is exhausted.

No external libraries are used.
"""

from __future__ import annotations

from decimal import Decimal, getcontext, ROUND_HALF_UP
from fractions import Fraction
from typing import Dict, Tuple


def prob_pair_in_first_two() -> Fraction:
    """Probability that the 2nd card matches the 1st by rank."""
    # After drawing any first card, 3 of the remaining 51 cards share its rank.
    return Fraction(3, 51)


State = Tuple[int, int, int, int, int, int]
# (a0,a1,a2,a3,a4,c) where aj = number of ranks with j remaining copies,
# and c is the remaining copies of the most recently drawn rank.


def expected_cards_drawn() -> Fraction:
    """Return E[N] exactly as a Fraction."""
    # After 1 draw:
    # - one rank has 3 remaining cards
    # - twelve ranks have 4 remaining cards
    # last drawn rank has c=3 remaining
    init: State = (0, 0, 0, 1, 12, 3)

    states: Dict[State, Fraction] = {init: Fraction(1, 1)}

    # S[i] = P(no consecutive pair in the first i draws)
    # We only need S[1..51] for E[N] = 1 + sum_{i=1}^{51} S[i].
    S = [Fraction(0, 1)] * 52
    S[1] = Fraction(1, 1)

    # Build S[2]..S[51]
    for drawn in range(1, 51):
        remaining_cards = 52 - drawn
        next_states: Dict[State, Fraction] = {}

        for (a0, a1, a2, a3, a4, c), p in states.items():
            # Transition by drawing a different rank than the last one.
            # Choose a rank with r remaining copies (r>=1), excluding the last rank.
            a = [a0, a1, a2, a3, a4]
            for r in (1, 2, 3, 4):
                eligible_ranks = a[r] - (1 if r == c else 0)
                if eligible_ranks <= 0:
                    continue

                # Probability to draw any of those ranks:
                prob_draw = Fraction(eligible_ranks * r, remaining_cards)

                b = a[:]  # updated histogram
                b[r] -= 1
                b[r - 1] += 1
                c2 = r - 1
                ns: State = (b[0], b[1], b[2], b[3], b[4], c2)
                next_states[ns] = next_states.get(ns, Fraction(0, 1)) + p * prob_draw

        states = next_states
        S[drawn + 1] = sum(states.values(), Fraction(0, 1))

    # Expected value from survival probabilities.
    exp_val = Fraction(1, 1) + sum(S[1:52], Fraction(0, 1))
    return exp_val


def format_rounded_8(x: Fraction) -> str:
    """Round to 8 places after the decimal point (as a string)."""
    getcontext().prec = 80
    d = Decimal(x.numerator) / Decimal(x.denominator)
    return str(d.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP))


def main() -> None:
    # Test value from the problem statement.
    assert prob_pair_in_first_two() == Fraction(1, 17)

    ans = expected_cards_drawn()

    # Light sanity checks that don't reveal the final answer.
    assert Fraction(2, 1) <= ans <= Fraction(52, 1)

    print(format_rounded_8(ans))


if __name__ == "__main__":
    main()
