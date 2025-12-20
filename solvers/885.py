#!/usr/bin/env python3
"""Project Euler 885: Sorted Digits

For a positive integer d, define f(d) as the number formed by sorting the digits
of d in ascending order and deleting any zeros.

Let S(n) be the sum of f(d) for all positive integers d with at most n digits.
Compute S(18) modulo 1123455689.

This implementation uses the identity:

    S(n) = (1/9) * sum_{k=1..9} (10 + 9k)^n  - 10^n

and evaluates it with modular exponentiation.

No external libraries are used.
"""

from typing import Optional, Tuple

MOD = 1123455689


def _egcd(a: int, b: int) -> Tuple[int, int, int]:
    """Extended Euclidean algorithm.

    Returns (g, x, y) where g=gcd(a, b) and a*x + b*y = g.
    """
    if b == 0:
        return a, 1, 0
    g, x1, y1 = _egcd(b, a % b)
    return g, y1, x1 - (a // b) * y1


def modinv(a: int, m: int) -> int:
    """Modular inverse of a modulo m (requires gcd(a, m) == 1)."""
    g, x, _ = _egcd(a, m)
    if g != 1:
        raise ValueError("inverse does not exist")
    return x % m


def S_closed_form(n: int, mod: Optional[int] = None) -> int:
    """Compute S(n). If mod is given, returns S(n) modulo mod."""
    if n < 0:
        raise ValueError("n must be nonnegative")

    # Numbers with <= n digits can be represented as length-n digit strings
    # (allowing leading zeros). Removing zeros in f(d) makes this representation
    # compatible, and the all-zero string contributes 0.

    if mod is None:
        total = sum((10 + 9 * k) ** n for k in range(1, 10)) - 9 * (10**n)
        return total // 9

    inv9 = modinv(9, mod)
    total = (
        sum(pow(10 + 9 * k, n, mod) for k in range(1, 10)) - (9 * pow(10, n, mod))
    ) % mod
    return (total * inv9) % mod


def main() -> None:
    # Test values given in the problem statement.
    assert S_closed_form(1) == 45
    assert S_closed_form(5) == 1543545675

    print(S_closed_form(18, MOD))


if __name__ == "__main__":
    main()
