#!/usr/bin/env python
from __future__ import annotations


def is_prime(n: int) -> bool:
    """
    @formalsw: Returns
    """
    if n < 2:
        return False
    small_primes: list[int] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False

    # Miller-Rabin deterministic for 64-bit using known bases
    # (Overkill for this problem where n ~ 7e8, but fast and safe.)
    d = n - 1
    s = 0
    while d % 2 == 0:
        s += 1
        d //= 2

    def check(a: int) -> bool:
        if a % n == 0:
            return True
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            return True
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                return True
        return False

    for a in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if not check(a):
            return False
    return True


def solve() -> int:
    """
    @formalsw: Returns
    @formalsw: Py from project_euler_statements.p58 import naive_default
    @formalsw: Equiv(naive_default)
    """
    # Start with side length 1
    prime_count = 0
    total_diag = 1

    k = 0
    while True:
        k += 1
        side = 2 * k + 1
        step = side - 1
        sq = side * side

        # Four corners are: sq, sq-step, sq-2*step, sq-3*step.
        # sq is an odd square (composite for side>1), so only test the other three.
        corners = (sq - step, sq - 2 * step, sq - 3 * step)
        prime_count += sum(1 for x in corners if is_prime(x))
        total_diag += 4

        # first time the ratio falls below 10%
        if prime_count * 10 < total_diag:
            return side


def _prime_ratio_counts_for_side(side: int) -> tuple[int, int]:
    """(prime_count_on_diagonals, total_numbers_on_diagonals) for odd side."""
    assert side % 2 == 1 and side >= 1
    prime_count = 0
    total_diag = 1
    kmax = (side - 1) // 2
    for k in range(1, kmax + 1):
        s = 2 * k + 1
        step = s - 1
        sq = s * s
        corners = (sq - step, sq - 2 * step, sq - 3 * step)
        prime_count += sum(1 for x in corners if is_prime(x))
        total_diag += 4
    return prime_count, total_diag


if __name__ == "__main__":
    # Problem statement check: for side length 7, ratio is 8/13
    assert _prime_ratio_counts_for_side(7) == (8, 13)

    # Another small sanity check: side length 3 has primes 3,5,7 on diagonals => 3/5
    assert _prime_ratio_counts_for_side(3) == (3, 5)

    print(solve())
