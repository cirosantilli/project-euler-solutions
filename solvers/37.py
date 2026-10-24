#!/usr/bin/env python
from math import isqrt


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for divisor in range(3, isqrt(n) + 1, 2):
        if n % divisor == 0:
            return False
    return True


def is_left_truncatable(n: int, base: int) -> bool:
    power = base
    while power <= n:
        if not is_prime(n % power):
            return False
        power *= base
    return True


def solve(base: int = 10) -> int:
    """
    Sum primes with at least two digits that remain prime under either truncation.

    @formalsw: Requires(base >= 2)
    @formalsw: Py from project_euler_statements.p37 import naive
    @formalsw: Equiv(naive)
    """
    frontier = [n for n in range(2, base) if is_prime(n)]
    total = 0
    while frontier:
        following = []
        for prefix in frontier:
            for digit in range(1, base):
                candidate = base * prefix + digit
                if is_prime(candidate):
                    following.append(candidate)
                    # All right truncations are prime by construction.
                    if is_left_truncatable(candidate, base):
                        total += candidate
        frontier = following
    return total


if __name__ == "__main__":
    assert is_left_truncatable(3797, 10)
    print(solve())
