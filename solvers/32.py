#!/usr/bin/env python

from math import isqrt


def is_pandigital(s: str, width: int = 9) -> bool:
    if len(s) != width:
        return False
    mask = 0
    for ch in s:
        digit = ord(ch) - 48
        if digit < 1:
            return False
        if digit > width:
            return False
        bit = 1 << digit
        if mask & bit:
            return False
        mask |= bit
    return True


def solve(width: int = 9) -> int:
    """
    Sum distinct products whose two positive factors and product use digits 1..width once.

    @formalsw: Py from project_euler_statements.p32 import naive
    @formalsw: Returns
    @formalsw: Equiv(naive)
    """
    if width < 3:
        return 0
    if width > 9:
        return 0
    product_width = width // 2
    lower = 10 ** (product_width - 1)
    upper = 10 ** product_width - 1
    products: set[int] = set()
    for a in range(1, isqrt(upper) + 1):
        first_b = max(a, (lower + a - 1) // a)
        for b in range(first_b, upper // a + 1):
            product = a * b
            if is_pandigital(str(a) + str(b) + str(product), width):
                products.add(product)
    return sum(products)


if __name__ == "__main__":
    assert is_pandigital("391867254")
    print(solve())
