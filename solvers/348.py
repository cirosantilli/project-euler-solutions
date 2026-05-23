#!/usr/bin/env python
"""
Project Euler 348: Sum of a Square and a Cube

Generate palindromes in increasing order.  For each candidate n, count cubes
b^3 such that n - b^3 is a square a^2 with a,b > 1.
"""

from math import isqrt


def reverse_int(n: int) -> int:
    result = 0
    while n:
        result = 10 * result + n % 10
        n //= 10
    return result


def make_palindrome(prefix: int, odd_length: bool) -> int:
    if odd_length:
        return prefix * (10 ** (len(str(prefix)) - 1)) + reverse_int(prefix // 10)
    return prefix * (10 ** len(str(prefix))) + reverse_int(prefix)


def find_palindromes(target_count: int) -> list[int]:
    found: list[int] = []
    cubes: list[int] = []
    next_cube_base = 2
    digits = 1

    while len(found) < target_count:
        half_digits = (digits + 1) // 2
        start = 1 if half_digits == 1 else 10 ** (half_digits - 1)
        stop = 10**half_digits
        odd_length = digits % 2 == 1

        for prefix in range(start, stop):
            n = make_palindrome(prefix, odd_length)
            if n < 12:
                continue

            while next_cube_base**3 <= n - 4:
                cubes.append(next_cube_base**3)
                next_cube_base += 1

            representations = 0
            for cube in cubes:
                if cube + 4 > n:
                    break
                rem = n - cube
                root = isqrt(rem)
                if root * root == rem:
                    representations += 1
                    if representations > 4:
                        break

            if representations == 4:
                found.append(n)
                if len(found) == target_count:
                    break

        digits += 1

    return found


def solve(target_count: int = 5) -> int:
    return sum(find_palindromes(target_count))


def main() -> None:
    assert find_palindromes(1)[0] == 5229225
    print(solve())


if __name__ == "__main__":
    main()
