#!/usr/bin/env python
import itertools


def is_pandigital(number, base, all_bits):
    used = 0

    while number:
        digit = number % base
        used |= 1 << digit
        if used == all_bits:
            return True
        number //= base

    return used == all_bits


def is_pandigital_base8(number):
    used = 0

    while number:
        used |= 1 << (number & 7)
        if used == 0xFF:
            return True
        number >>= 3

    return used == 0xFF


def solve(base=12, num_results=10):
    digits = tuple(range(base))
    check_bases = list(range(base - 1, 1, -1))
    if 8 in check_bases:
        check_bases.remove(8)
        check_bases.insert(0, 8)
    all_bits = [0] * base
    for b in check_bases:
        all_bits[b] = (1 << b) - 1

    num_found = 0
    total = 0

    for first_digit in range(1, base):
        remaining = digits[:first_digit] + digits[first_digit + 1 :]
        for tail in itertools.permutations(remaining):
            current = first_digit
            for digit in tail:
                current = current * base + digit

            is_good = True
            for b in check_bases:
                if b == 8:
                    if not is_pandigital_base8(current):
                        is_good = False
                        break
                elif not is_pandigital(current, b, all_bits[b]):
                    is_good = False
                    break

            if is_good:
                total += current
                num_found += 1
                if num_found == num_results:
                    return total

    return total


def main():
    assert solve(10, 10) == 20319792309
    print(solve())


if __name__ == "__main__":
    main()
