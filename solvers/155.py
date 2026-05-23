#!/usr/bin/env python
"""
Project Euler 155: Counting Capacitor Circuits

Store each capacitance exactly as a reduced numerator/denominator pair.  The
reachable values for a fixed number of unit capacitors are closed under
reciprocal, so it is enough to generate x + y and then insert its reciprocal.
"""

from math import gcd


Capacitance = tuple[int, int]


def add_reduced(a: Capacitance, b: Capacitance) -> Capacitance:
    num = a[0] * b[1] + b[0] * a[1]
    den = a[1] * b[1]
    g = gcd(num, den)
    return num // g, den // g


def solve(limit: int) -> int:
    exact: list[set[Capacitance]] = [set() for _ in range(limit + 1)]
    exact[1].add((1, 1))

    all_values: set[Capacitance] = {(1, 1)}

    for total_size in range(2, limit + 1):
        current: set[Capacitance] = set()

        for left_size in range(1, total_size // 2 + 1):
            right_size = total_size - left_size
            left_values = list(exact[left_size])
            right_values = list(exact[right_size])

            if left_size == right_size:
                pairs = (
                    (left_values[i], left_values[j])
                    for i in range(len(left_values))
                    for j in range(i, len(left_values))
                )
            else:
                pairs = (
                    (left, right) for left in left_values for right in right_values
                )

            for left, right in pairs:
                value = add_reduced(left, right)
                current.add(value)
                current.add((value[1], value[0]))

        exact[total_size] = current
        all_values.update(current)

    return len(all_values)


def main() -> None:
    assert solve(1) == 1
    assert solve(2) == 3
    assert solve(3) == 7
    print(solve(18))


if __name__ == "__main__":
    main()
