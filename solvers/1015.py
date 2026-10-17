#!/usr/bin/env python


def grundy_values(n: int) -> list[int]:
    """Nimbers for removing adjacent pairs from an alternating block."""
    values = [0] * (n + 1)
    for length in range(2, n + 1):
        options = {
            values[left] ^ values[length - 2 - left]
            for left in range((length - 2) // 2 + 1)
        }
        value = 0
        while value in options:
            value += 1
        values[length] = value
    return values


def balanced_sum(values: list[int], character: int, m: int) -> int:
    """Signed number of balanced rows of length 2*m, for one XOR character."""
    signs = [1 - 2 * ((character & value).bit_count() % 2) for value in values]
    # E(x) records even blocks; z*T(x), with x=z^2, records odd blocks.
    even = [0] + [signs[2 * i] for i in range(1, m + 1)]
    odd = [signs[2 * i + 1] for i in range(m)]
    numerator = [1] + [0] * m
    denominator = [1] + [0] * m
    for degree in range(1, m + 1):
        common = sum(even[i] * even[degree - i] for i in range(1, degree))
        common -= sum(odd[i] * odd[degree - 1 - i] for i in range(degree))
        numerator[degree] = common + 2 * even[degree]
        denominator[degree] = common - 2 * even[degree]

    # S(x)^2 = numerator(x)/denominator(x). Keep exact integers: dividing
    # by two in a ring modulo 10^9 would lose information.
    quotient = [1] + [0] * m
    root = [1] + [0] * m
    for degree in range(1, m + 1):
        quotient[degree] = numerator[degree] - sum(
            denominator[i] * quotient[degree - i] for i in range(1, degree + 1)
        )
        twice = quotient[degree] - sum(
            root[i] * root[degree - i] for i in range(1, degree)
        )
        assert twice % 2 == 0
        root[degree] = twice // 2
    return root[m]


def solve(n: int, modulus: int = 10**9) -> int:
    """Count rows of n pins which are lost by either starting player."""
    if n % 2:
        return 0
    values = grundy_values(n)
    characters = 1 << max(values).bit_length()
    total = sum(balanced_sum(values, h, n // 2) for h in range(characters))
    assert total % characters == 0
    return total // characters % modulus


if __name__ == "__main__":
    assert solve(0) == 1
    assert solve(6) == 8
    assert solve(20) == 49844
    print(f"{solve(1000):09d}")
