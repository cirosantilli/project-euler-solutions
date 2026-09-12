#!/usr/bin/env python

from __future__ import annotations


def largest_same_digits(a: int, b: int) -> int:
    """Return F(a, b)."""
    if b >= 3 * a:
        return 0

    # For b >= 2a, only a two-digit representation can work.
    if b >= 2 * a:
        weight = b - 2 * a
        if weight == 0:
            return a * (a - 1)
        leading_digit = (a - 1) // weight
        return leading_digit * (b - a)

    # Let w_i = b^i - 2a^i. There is one sign change. If r is the
    # first positive index, every non-zero solution has exactly r+1 digits.
    r = 1
    a_power = a
    b_power = b
    while b_power < 2 * a_power:
        r += 1
        a_power *= a
        b_power *= b
    positive_weight = b_power - 2 * a_power

    # For i < r write v_i = -w_i = 2a^i - b^i > 0.
    negative_weights = [0] * r
    capacities = [0] * r
    a_power = a
    b_power = b
    for i in range(1, r):
        negative_weights[i] = 2 * a_power - b_power
        capacities[i] = capacities[i - 1] + (a - 1) * negative_weights[i]
        a_power *= a
        b_power *= b

    def best_lower_digits(i: int, low: int, high: int) -> tuple[int, ...] | None:
        """Largest d_i...d_1 with sum d_j v_j in [low, high]."""
        if high < 0 or low > capacities[i]:
            return None
        low = max(low, 0)
        high = min(high, capacities[i])
        if low > high:
            return None
        if i == 0:
            return ()

        weight = negative_weights[i]
        lower_capacity = capacities[i - 1]
        first = max(0, (low - lower_capacity + weight - 1) // weight)
        last = min(a - 1, high // weight)

        for digit in range(last, first - 1, -1):
            rest = best_lower_digits(
                i - 1,
                low - digit * weight,
                high - digit * weight,
            )
            if rest is not None:
                return (digit,) + rest
        return None

    # d_0 = d_r w_r - sum_{i<r} d_i v_i must lie in [0, a-1].
    for leading_digit in range(a - 1, 0, -1):
        target = leading_digit * positive_weight
        lower_digits = best_lower_digits(r - 1, target - (a - 1), target)
        if lower_digits is None:
            continue

        negative_sum = sum(
            digit * negative_weights[i]
            for digit, i in zip(lower_digits, range(r - 1, 0, -1))
        )
        final_digit = target - negative_sum

        n = leading_digit
        for digit in lower_digits:
            n = n * a + digit
        return n * a + final_digit

    return 0


def g(a: int) -> int:
    # F(a,b)=0 for b >= 3a.
    return sum(largest_same_digits(a, b) for b in range(a + 1, 3 * a))


def main() -> None:
    assert largest_same_digits(3, 4) == 53
    assert largest_same_digits(9, 10) == 8152650
    assert g(3) == 72
    print(sum(g(a) for a in range(2, 21)))


if __name__ == "__main__":
    main()
