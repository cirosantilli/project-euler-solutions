#!/usr/bin/env python
from __future__ import annotations


def terminal_value(a: int, b: int) -> int:
    while a != 1 and b != 1:
        if a < b:
            a, b = b, a
        a //= b
    return max(a, b)


def euclidean_sum(limit: int) -> int:
    last = limit - 1
    # In the upper triangle, b // a = 1 gives f(a, b) = a.
    upper = sum(a * min(a, limit - a) for a in range(1, limit))
    for a in range(1, last // 2 + 1):
        for quotient in range(2, last // a + 1):
            multiplicity = min(a, limit - a * quotient)
            upper += multiplicity * terminal_value(a, quotient)
    # Reflect the triangle and subtract its diagonal, where f(a, a) = a.
    return 2 * upper - last * limit // 2


def main() -> None:
    assert terminal_value(123, 456) == 3
    assert euclidean_sum(10) == 343
    assert euclidean_sum(100) == 269288
    print(euclidean_sum(3_000_000))


if __name__ == "__main__":
    main()
