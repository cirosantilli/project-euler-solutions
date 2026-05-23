#!/usr/bin/env python
"""Adapted from: https://github.com/stbrumme/euler/blob/b426763514558c3b39f2ec507f271d322088d28a/euler-0510.cpp"""

from math import gcd


def triangle(n):
    nn = n
    return nn * (nn + 1) // 2


def evaluate(limit):
    result = 0
    n = 1
    while (n + 1) * (n + 1) * n * n <= limit:
        n2 = n * n

        for m in range(1, n + 1):
            if gcd(m, n) != 1:
                continue

            m2 = m * m
            sum2 = (m + n) * (m + n)
            b0 = sum2 * n2
            if b0 > limit:
                break

            a0 = sum2 * m2
            c0 = m2 * n2
            multiples = limit // b0
            result += (a0 + b0 + c0) * triangle(multiples)

        n += 1
    return result


def main():
    assert evaluate(5) == 9
    assert evaluate(100) == 3072
    print(evaluate(1000000000))


if __name__ == "__main__":
    main()
