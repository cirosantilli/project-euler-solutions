#!/usr/bin/env python

from math import acos, cos, pi, sin, tan


def compute(n: int) -> float:
    theta = pi / n
    tangent = tan(theta)

    branch = 0
    for k in range(n + 1):
        value = sin(k * theta) - (k + n) * tangent * cos(k * theta)
        if value >= 0:
            branch = k - 1
            break

    argument = 2 * sin(branch * theta) / ((branch + n) * tangent) - cos(branch * theta)
    argument = min(1.0, max(-1.0, argument))
    alpha = (branch * theta + acos(argument)) / 2
    return round(1 / cos(alpha), 8)


if __name__ == "__main__":
    print(compute(6))
