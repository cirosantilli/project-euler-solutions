#!/usr/bin/env python3
"""
Project Euler 905 - Now I Know

Compute:
    sum_{a=1..7} sum_{b=1..19} F(a^b, b^a, a^b + b^a)

Game definition:
- A, B, C wear positive integers.
- Exactly one of the three equals the sum of the other two.
- They see the other two hats, not their own.
- Starting with A and proceeding cyclically (A,B,C,A,B,C,...),
  each says either:
    "I don't know my number"
  or
    "Now I know my number!"
  which ends the game.

Let F(A,B,C) be the number of turns until the game ends (inclusive).

This solution derives a fast exact method by reducing the epistemic
reasoning to a 3-state Euclidean subtraction process, then accelerating
large quotients using division batching.
"""


def _k_fast(state: int, a: int, b: int) -> int:
    """
    Return k_state(a,b):

    We normalize every valid triple into one of three 'states':
      state 0: A is the sum-holder, triple = (a+b, a, b)  (B=a, C=b)
      state 1: B is the sum-holder, triple = (a, a+b, b)  (A=a, C=b)
      state 2: C is the sum-holder, triple = (a, b, a+b)  (A=a, B=b)

    k_state(a,b) counts how many *full cycles* (A,B,C) elapse before the
    sum-holder announces on their own turn. For state 2 (C sum-holder),
    k>=1 because C cannot speak before turn 3.

    The naive recursion subtracts one smaller number at a time, i.e. the
    subtractive Euclidean algorithm. We accelerate long runs by observing
    that many steps occur in pairs that subtract 2*min each time. We can
    batch those using integer division.
    """
    cost = 0

    while a != b:
        if state == 2:
            if a > b:
                # Paired cycle reduces 'a' by 2*b and returns to state 2
                p = (a - 1) // (2 * b)
                if p:
                    cost += p
                    a -= 2 * b * p
                    continue
                # One step
                cost += 1
                # (2, a, b) with a>b -> (0, b, a-b)
                state = 0
                a, b = b, a - b
            else:
                # b > a
                p = (b - 1) // (2 * a)
                if p:
                    cost += p
                    b -= 2 * a * p
                    continue
                cost += 1
                # (2, a, b) with b>a -> (1, a, b-a)
                state = 1
                b = b - a

        elif state == 0:
            if a > b:
                # Paired cycle reduces 'a' by 2*b and returns to state 0
                p = (a - 1) // (2 * b)
                if p:
                    cost += p
                    a -= 2 * b * p
                    continue
                cost += 1
                # (0, a, b) with a>b -> (1, a-b, b)
                state = 1
                a = a - b
            else:
                # b > a
                # (0, a, b) -> (2, b-a, a)
                state = 2
                a, b = b - a, a

        else:
            # state == 1: all transitions cost 0
            if a > b:
                # (1, a, b) -> (0, a-b, b)
                state = 0
                a = a - b
            else:
                # b > a
                # (1, a, b) -> (2, a, b-a)
                state = 2
                b = b - a

    # Now a==b, apply base:
    # state 0 or 1: k=0
    # state 2: k=1
    return cost + (1 if state == 2 else 0)


def F(A: int, B: int, C: int) -> int:
    """
    Compute the number of turns until someone announces.

    We first determine which player holds the sum (the largest number).
    Then compute k with _k_fast and translate back to turn count:

      state 0: ends on A's turn => turns = 3*k + 1
      state 1: ends on B's turn => turns = 3*k + 2
      state 2: ends on C's turn => turns = 3*k

    Note: In state 2, k already includes the unavoidable '1' when a==b.
    """
    if A == B + C:
        # A is sum-holder, pair is (B,C)
        k = _k_fast(0, B, C)
        return 3 * k + 1
    if B == A + C:
        # B is sum-holder, pair is (A,C)
        k = _k_fast(1, A, C)
        return 3 * k + 2
    if C == A + B:
        # C is sum-holder, pair is (A,B)
        k = _k_fast(2, A, B)
        return 3 * k
    raise ValueError("Invalid triple: none equals sum of the other two.")


def main() -> None:
    # Problem statement test values
    assert F(2, 1, 1) == 1
    assert F(2, 7, 5) == 5

    total = 0
    for a in range(1, 8):
        for b in range(1, 20):
            A = pow(a, b)
            B = pow(b, a)
            total += F(A, B, A + B)

    print(total)


if __name__ == "__main__":
    main()
