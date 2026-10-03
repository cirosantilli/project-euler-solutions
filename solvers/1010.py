#!/usr/bin/env python
from __future__ import annotations

from array import array
from math import factorial, isqrt


MODULUS = 1234567891


def endpoint_count(positions: tuple[int, ...]) -> int:
    shape = [position - i for i, position in enumerate(positions, 1)][::-1]
    denominator = 1
    for i, width in enumerate(shape):
        for j in range(width):
            leg = sum(other > j for other in shape[i + 1 :])
            denominator *= width - j + leg
    return factorial(sum(shape)) // denominator


def march_product(penguins: int, steps: int, modulus: int = MODULUS) -> int:
    # The stated modulus is prime, and greater than every factorial/hook factor.
    assert modulus > steps
    assert all(modulus % d for d in range(2, isqrt(modulus) + 1))
    period = modulus - 1
    m = min(penguins, steps)
    if steps == 0:
        return 1
    baseline = m * (m - 1) // 2
    bound = steps + baseline
    # partitions[j][k] counts partitions of k with at most j parts.
    partitions = [array("I", [0]) * (bound + 1)]
    partitions[0][0] = 1
    for j in range(1, m + 1):
        row = partitions[-1][:]
        for k in range(j, bound + 1):
            row[k] = (row[k] + row[k - j]) % period
        partitions.append(row)

    numerator = pow(factorial(steps) % modulus, partitions[m][steps], modulus)
    denominator = 1
    for hook in range(1, steps + 1):
        exponent = 0
        # Sum marked low beads over sizes steps-hook, steps-2*hook, ... .
        for removed in range(1, m + 1):
            j = m - removed
            row = partitions[j]
            shift = baseline - j * (j - 1) // 2
            subtotal = 0
            for size in range(steps - hook, -1, -hook):
                start = size + shift
                stop = max(-1, start - removed * hook)
                for index in range(start, stop, -removed):
                    subtotal += row[index]
            if removed % 2:
                exponent += subtotal
            else:
                exponent -= subtotal
        denominator = denominator * pow(hook, exponent % period, modulus) % modulus
    return numerator * pow(denominator, -1, modulus) % modulus


def main() -> None:
    assert endpoint_count((1, 6)) == 1
    assert endpoint_count((2, 5)) == 3
    assert endpoint_count((3, 4)) == 2
    assert march_product(2, 4) == 6
    assert march_product(3, 6) == 180000
    assert march_product(5, 10) == 411456133
    print(march_product(150, 300))


if __name__ == "__main__":
    main()
