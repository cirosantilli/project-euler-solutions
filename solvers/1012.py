#!/usr/bin/env python
from __future__ import annotations

from collections.abc import Iterator


# Fixed-point arithmetic keeps substantially more precision than the output.
SCALE = 10**60
TOLERANCE = 10**24


def determinant(rows: list[list[int]]) -> int:
    a, b, c = rows
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def interval_strategy(first: int, last: int) -> list[int]:
    length = last - first + 1
    assert length >= 3 and length % 2 == 1
    weight = 2 * first - 1
    # x[t] = a*u[t] + b*v[t] + d*c[t] + z[t], where
    # d = sum 2*t*x[t].  The indifference equations imply
    # (weight+2*t+4)x[t+2] = (weight+2*t)x[t] + 2*x[t+1]
    #                            + 2*d - 4*(t+1).
    u = [SCALE, 0]
    v = [0, SCALE]
    c = [0, 0]
    z = [0, 0]
    for t in range(length - 2):
        w = weight + 2 * t
        denominator = w + 4
        u.append((w * u[-2] + 2 * u[-1]) // denominator)
        v.append((w * v[-2] + 2 * v[-1]) // denominator)
        c.append((w * c[-2] + 2 * c[-1] + 2 * SCALE) // denominator)
        z.append((w * z[-2] + 2 * z[-1] - 4 * (t + 1) * SCALE) // denominator)

    # Enforce normalization, the definition of d, and the first row's payoff.
    columns = []
    for values in (u, v, c, z):
        moment = 0
        payoff = 0
        for t in range(1, length):
            moment += 2 * t * values[t]
            payoff += (weight if t % 2 else -(weight + 2 * t)) * values[t]
        columns.append([sum(values), moment, payoff])
    columns[2][1] -= SCALE
    matrix = [[columns[j][i] for j in range(3)] for i in range(3)]
    rhs = [SCALE - columns[3][0], -columns[3][1], -columns[3][2]]
    denominator = determinant(matrix)
    assert denominator != 0
    coefficients = []
    for j in range(3):
        replaced = [row[:] for row in matrix]
        for i in range(3):
            replaced[i][j] = rhs[i]
        coefficients.append(determinant(replaced) * SCALE // denominator)
    a, b, d = coefficients
    strategy = [
        (a * u[t] + b * v[t] + d * c[t]) // SCALE + z[t]
        for t in range(length)
    ]
    first_payoff = sum(
        (weight if t % 2 else -(weight + 2 * t)) * strategy[t]
        for t in range(1, length)
    )
    last_payoff = sum(
        (-(weight + 2 * t) if t % 2 else 2 * last - 1) * strategy[t]
        for t in range(length - 1)
    )
    assert abs(first_payoff) < TOLERANCE
    assert abs(last_payoff) < TOLERANCE
    return strategy


def probabilities(limit: int) -> Iterator[int]:
    first = 1
    last = 2
    strategy: list[int] = []
    length = 3
    for n in range(3, limit + 1):
        if strategy and last == n - 1:
            # If the newly added option cannot beat the previous equilibrium,
            # that equilibrium remains valid and the new probability is zero.
            odd_mass = sum(strategy[1::2])
            payoff = 2 * length * odd_mass - (2 * first - 1) * strategy[0]
            if payoff <= 0:
                yield 0
                continue

        while length <= n:
            first = n - length + 1
            strategy = interval_strategy(first, n)
            even_mass = sum(strategy[::2])
            # All earlier excluded rows follow from the closest one and x[0].
            outside = (2 * n - 1) * strategy[-1] - 2 * length * even_mass
            if min(strategy) >= 0 and (first == 1 or outside <= 0):
                break
            length += 2
        else:
            raise ArithmeticError("No valid equilibrium interval found")

        assert abs(sum(strategy) - SCALE) < TOLERANCE
        last = n
        yield strategy[-1]


def rounded(value: int) -> str:
    unit = SCALE // 10**10
    digits = (value + unit // 2) // unit
    return f"{digits // 10**10}.{digits % 10**10:010d}"


def equilibrium_sum(limit: int) -> str:
    return rounded(sum(probabilities(limit)))


def main() -> None:
    sample = list(probabilities(10))
    assert abs(sample[0] * 9 - SCALE) < TOLERANCE
    assert abs(sample[1] * 5 - SCALE) < TOLERANCE
    assert rounded(sample[-1]) == "0.0479638009"
    assert equilibrium_sum(10) == "1.1546112276"
    assert equilibrium_sum(100) == "4.8779925686"
    print(equilibrium_sum(100_000))


if __name__ == "__main__":
    main()
