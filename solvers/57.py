#!/usr/bin/env python


def solve(limit: int) -> int:
    """
    Count among the first `limit` convergents of sqrt(2) (as in the problem's expansions)
    how many have a numerator with more digits than the denominator.
    """
    # First expansion: 3/2
    p, q = 3, 2
    cnt = 0

    for _ in range(1, limit + 1):
        if len(str(p)) > len(str(q)):
            cnt += 1
        # Next convergent recurrence for sqrt(2) continued fraction:
        # p' = p + 2q
        # q' = p + q
        p, q = p + 2 * q, p + q

    return cnt


if __name__ == "__main__":
    # The 8th expansion 1393/985 is the first where numerator has more digits.
    assert solve(7) == 0
    assert solve(8) == 1
    print(solve(1000))
