#!/usr/bin/env python3
"""Project Euler 845: Prime Digit Sum

Let D(n) be the n-th positive integer whose sum of digits is prime.
Compute D(10^16).

This solution uses digit-sum dynamic programming to count how many numbers of a
fixed length have prime digit sum, then constructs the k-th such number
lexicographically (which matches numeric order within a fixed length).

No external libraries are used.
"""

from __future__ import annotations

from math import isqrt


def sieve_is_prime(limit: int) -> list[bool]:
    """Return boolean primality table for 0..limit inclusive."""
    if limit < 0:
        return []
    is_prime = [True] * (limit + 1)
    if limit >= 0:
        is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for p in range(2, isqrt(limit) + 1):
        if is_prime[p]:
            start = p * p
            step = p
            is_prime[start : limit + 1 : step] = [False] * (
                ((limit - start) // step) + 1
            )
    return is_prime


def build_digit_sum_counts(max_len: int) -> list[list[int]]:
    """counts[L][s] = # of length-L digit strings (leading zeros allowed) with digit-sum s."""
    max_sum = 9 * max_len
    counts = [[0] * (max_sum + 1) for _ in range(max_len + 1)]
    counts[0][0] = 1
    for L in range(1, max_len + 1):
        for s in range(0, 9 * L + 1):
            total = 0
            # last digit is d, previous sum is s-d
            for d in range(10):
                if s - d < 0:
                    break
                total += counts[L - 1][s - d]
            counts[L][s] = total
    return counts


def count_len_prime_digit_sum(
    length: int, counts: list[list[int]], is_prime: list[bool]
) -> int:
    """How many positive integers with exactly `length` digits have prime digit sum?"""
    if length <= 0:
        return 0
    if length == 1:
        # 1-digit positive integers are 1..9
        return sum(1 for d in range(1, 10) if is_prime[d])

    total = 0
    rem = length - 1
    for first in range(1, 10):
        for tail_sum in range(0, 9 * rem + 1):
            if is_prime[first + tail_sum]:
                total += counts[rem][tail_sum]
    return total


def kth_len_number(
    length: int, k: int, counts: list[list[int]], is_prime: list[bool]
) -> int:
    """Return the k-th (1-indexed) length-digit positive integer with prime digit sum."""
    if length <= 0:
        raise ValueError("length must be positive")
    if k <= 0:
        raise ValueError("k must be positive (1-indexed)")

    digits: list[str] = []
    prefix_sum = 0

    for pos in range(length):
        rem = length - pos - 1
        start = 1 if pos == 0 else 0

        for d in range(start, 10):
            cnt = 0
            base = prefix_sum + d
            # choose rem digits (with leading zeros) so that total sum is prime
            for tail_sum in range(0, 9 * rem + 1):
                if is_prime[base + tail_sum]:
                    cnt += counts[rem][tail_sum]

            if k > cnt:
                k -= cnt
            else:
                digits.append(str(d))
                prefix_sum += d
                break
        else:
            # If we get here, k was too large for this length.
            raise ValueError("k exceeds the number of valid numbers for this length")

    return int("".join(digits))


def D(n: int) -> int:
    """Compute D(n): the n-th positive integer whose digit sum is prime."""
    if n <= 0:
        raise ValueError("n must be positive")

    # For n=10^16 the answer fits well within <= 20 digits; we keep a small safety margin.
    # (Digit sums only matter up to 9*max_len.)
    max_len = 25
    is_prime = sieve_is_prime(9 * max_len)
    counts = build_digit_sum_counts(max_len)

    remaining = n
    for length in range(1, max_len + 1):
        cnt_len = count_len_prime_digit_sum(length, counts, is_prime)
        if remaining > cnt_len:
            remaining -= cnt_len
        else:
            return kth_len_number(length, remaining, counts, is_prime)

    raise RuntimeError("max_len too small; increase it")


def main() -> None:
    # Test values given in the problem statement:
    assert D(61) == 157
    assert D(10**8) == 403539364

    # Required output:
    print(D(10**16))


if __name__ == "__main__":
    main()
