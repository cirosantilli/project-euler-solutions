#!/usr/bin/env python

def digit_at(n: int) -> int:
    """
    Return the nth digit (1-indexed) of the fractional part of Champernowne's constant:
    0.12345678910111213...
    """
    assert n >= 1

    length = 1  # number of digits per integer in current block
    start = 1  # first integer in current block
    count = 9  # how many integers in current block

    while True:
        block_digits = count * length
        if n > block_digits:
            n -= block_digits
            length += 1
            start *= 10
            count *= 10
        else:
            # n lies within this block
            idx = (n - 1) // length  # which number in the block (0-based)
            pos = (n - 1) % length  # which digit in that number (0-based)
            num = start + idx
            return int(str(num)[pos])


def solve(positions: tuple[int, ...]) -> int:
    """
    Multiply the digits at the supplied positive one-based positions.

    @formalsw: Py from project_euler_statements.p40 import naive
    @formalsw: Requires(all(position > 0 for position in positions))
    @formalsw: Returns
    @formalsw: Equiv(naive)
    """
    product = 1
    for position in positions:
        product *= digit_at(position)
    return product


if __name__ == "__main__":
    # Given example in the statement
    assert digit_at(12) == 1

    # Known digits for Euler #40 (sanity checks)
    assert digit_at(1) == 1
    assert digit_at(10) == 1
    assert digit_at(100) == 5
    assert digit_at(1000) == 3
    assert digit_at(10000) == 7
    assert digit_at(100000) == 2
    assert digit_at(1000000) == 1

    print(solve((1, 10, 100, 1000, 10000, 100000, 1000000)))
