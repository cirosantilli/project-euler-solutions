#!/usr/bin/env python3
"""
Project Euler 655

Count positive base-10 palindromes < 10^32 that are divisible by 10_000_019.

Constraints for this solution:
- No external libraries (only the Python standard library).
- Single core / no multithreading.
"""

from __future__ import annotations

from array import array
from math import gcd


def is_palindrome(n: int) -> bool:
    s = str(n)
    return s == s[::-1]


def first_k_palindromic_multiples(mod: int, k: int, search_limit: int) -> list[int]:
    """Brute-force helper for small examples only."""
    out: list[int] = []
    x = mod
    while x <= search_limit and len(out) < k:
        if is_palindrome(x):
            out.append(x)
        x += mod
    return out


def _egcd(a: int, b: int) -> tuple[int, int, int]:
    """Return (g, x, y) with a*x + b*y = g = gcd(a,b)."""
    x0, y0, x1, y1 = 1, 0, 0, 1
    while b:
        q = a // b
        a, b = b, a - q * b
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    return a, x0, y0


def modinv(a: int, m: int) -> int:
    g, x, _ = _egcd(a, m)
    if g != 1:
        raise ValueError("inverse does not exist")
    return x % m


def count_palindromes_less_than_power10(max_len: int, mod: int) -> int:
    """
    Count positive palindromes with 1..max_len digits that are divisible by mod.

    Fast path assumes gcd(mod, 10) == 1 (true for the actual problem and the given example).
    """
    if max_len < 1:
        return 0
    if mod % 2 == 0 or mod % 5 == 0:
        raise ValueError("This method assumes gcd(mod, 10) == 1.")

    m = mod
    inv10 = modinv(10, m)

    # Precompute powers of 10 modulo m.
    pow10 = [1] * (max_len + 1)
    for i in range(1, max_len + 1):
        pow10[i] = (pow10[i - 1] * 10) % m

    def extend_all(dp: array, tmp: array, c: int) -> None:
        """
        Given dp for palindromic strings of length L (allowing leading zeros),
        overwrite dp with counts for length L+2 after adding a digit pair (d ... d)
        where d ranges over 0..9.

        Let c = (10^(L+1) + 1) mod m. Then for residues:
            next[r] = sum_{d=0..9} tmp[(r - d*c) mod m]
        where tmp is dp permuted by multiplication by 10.
        """
        # tmp[j] = dp[j * inv10 mod m] (permute by *10, since gcd(10,m)=1)
        src = 0
        for j in range(m):
            tmp[j] = dp[src]
            src += inv10
            if src >= m:
                src -= m

        # If c == 0 then the outer digits contribute 0 modulo m:
        # next[r] = 10 * tmp[r]
        if c == 0:
            for i in range(m):
                dp[i] = tmp[i] * 10
            return

        # If c is not invertible (only possible when m is not prime), fall back to a
        # direct O(10*m) update. This never triggers for the Euler modulus.
        if gcd(c, m) != 1:
            for r in range(m):
                s = 0
                base = r
                for d in range(10):
                    s += tmp[base]
                    base -= c
                    base %= m
                dp[r] = s
            return

        # Fast update: visit residues in the order 0, c, 2c, ... (mod m).
        # Along this cycle, the digit-sum is a cyclic sliding window of size 10.
        idx = (-10 * c) % m  # position for k = -10
        window = [0] * 10
        s = 0
        for t in range(10):
            v = tmp[idx]
            window[t] = v
            s += v
            idx += c
            if idx >= m:
                idx -= m

        idx = 0
        pos = 0
        for _ in range(m):
            v = tmp[idx]
            old = window[pos]
            s += v - old
            window[pos] = v
            pos += 1
            if pos == 10:
                pos = 0
            dp[idx] = s
            idx += c
            if idx >= m:
                idx -= m

    total = 0

    # Length 1: digits 1..9 (exclude 0).
    for d in range(1, 10):
        if d % m == 0:
            total += 1

    # Even lengths: start from empty string (length 0).
    dp = array("Q", [0]) * m
    tmp = array("Q", [0]) * m
    dp[0] = 1
    cur_len = 0
    while cur_len + 2 <= max_len:
        new_len = cur_len + 2
        c = (pow10[new_len - 1] + 1) % m
        prev_zero = dp[0]
        extend_all(dp, tmp, c)
        # Exclude outer digit 0: the removed contribution at remainder 0 is prev_zero.
        total += dp[0] - prev_zero
        cur_len = new_len

    # Odd lengths: start from a 1-digit center (0..9).
    dp = array("Q", [0]) * m
    tmp = array("Q", [0]) * m
    for d in range(10):
        dp[d % m] += 1

    cur_len = 1
    while cur_len + 2 <= max_len:
        new_len = cur_len + 2
        c = (pow10[new_len - 1] + 1) % m
        prev_zero = dp[0]
        extend_all(dp, tmp, c)
        total += dp[0] - prev_zero
        cur_len = new_len

    return total


def _run_problem_statement_asserts() -> None:
    # The three smallest palindromes divisible by 109.
    assert first_k_palindromic_multiples(109, 3, 20000) == [545, 5995, 15151]

    # There are nine palindromes less than 100000 divisible by 109.
    brute = 0
    for n in range(1, 100000):
        if n % 109 == 0 and is_palindrome(n):
            brute += 1
    assert brute == 9
    assert count_palindromes_less_than_power10(5, 109) == 9


def main() -> None:
    _run_problem_statement_asserts()
    print(count_palindromes_less_than_power10(32, 10000019))


if __name__ == "__main__":
    main()
