#!/usr/bin/env python
from __future__ import annotations

MOD = 1_000_000_007


def next_permutation_value(n: int) -> int:
    digits = [ord(c) - 48 for c in str(n)]

    i = len(digits) - 2
    while i >= 0 and digits[i] >= digits[i + 1]:
        i -= 1
    if i < 0:
        return 0

    j = len(digits) - 1
    while digits[j] <= digits[i]:
        j -= 1

    digits[i], digits[j] = digits[j], digits[i]
    lo = i + 1
    hi = len(digits) - 1
    while lo < hi:
        digits[lo], digits[hi] = digits[hi], digits[lo]
        lo += 1
        hi -= 1

    value = 0
    for digit in digits:
        value = value * 10 + digit
    return value


def B(n: int) -> int:
    return next_permutation_value(n)


def delta_mod(n: int) -> int:
    return (next_permutation_value(n) - n) % MOD


def T_bruteforce(limit: int) -> int:
    return sum(B(n * n) for n in range(1, limit + 1))


def square_sum_below_power10(k: int) -> int:
    n = pow(10, k, MOD)
    return n * ((n - 1) % MOD) % MOD * ((2 * n - 1) % MOD) % MOD * pow(6, MOD - 2, MOD) % MOD


def correction_sum(k: int) -> int:
    pow10 = [1]
    for _ in range(1, 2 * k + 3):
        pow10.append(pow10[-1] * 10)

    pow10_mod = [1]
    for _ in range(1, k + 1):
        pow10_mod.append((pow10_mod[-1] * 10) % MOD)

    def recurse(suffix: int, width: int, trailing_zeros: int) -> int:
        if width + trailing_zeros >= k:
            return delta_mod((suffix * pow10[trailing_zeros]) ** 2)

        square_suffix = (suffix * suffix) % pow10[width]
        left_digit = square_suffix // pow10[width - 1]
        next_place = pow10[width]
        next_modulus = pow10[width + 1]
        free_digits = k - width - trailing_zeros - 1
        completion_count = pow10_mod[free_digits]

        total = 0
        for digit in range(10):
            next_suffix = digit * next_place + suffix
            next_square = next_suffix * next_suffix
            new_digit = (next_square % next_modulus) // next_place

            if new_digit >= left_digit:
                total += recurse(next_suffix, width + 1, trailing_zeros)
            else:
                visible = (next_square % next_modulus) * pow10[2 * trailing_zeros]
                representative = pow10[width + 1 + 2 * trailing_zeros] + visible
                representative_delta = delta_mod(representative)

                if next_square < next_modulus:
                    total += delta_mod(visible)
                    total += (completion_count - 1) * representative_delta
                else:
                    total += completion_count * representative_delta

            total %= MOD

        return total

    total = 0
    for trailing_zeros in range(k):
        for last_digit in range(1, 10):
            total = (total + recurse(last_digit, 1, trailing_zeros)) % MOD
    return total


def solve(k: int = 16) -> int:
    return (square_sum_below_power10(k) + correction_sum(k)) % MOD


def main() -> None:
    assert B(245) == 254
    assert B(542) == 0
    assert T_bruteforce(10) == 270
    assert T_bruteforce(100) == 335316
    print(solve(16))


if __name__ == "__main__":
    main()
