#!/usr/bin/env python
def largest_palindrome_product(lo: int, hi: int) -> int:
    """
    Return the largest palindromic product of factors in [lo, hi].

    @formalsw: Returns
    """
    best = 0
    for a in range(hi, lo - 1, -1):
        # If even the maximum product with this 'a' can't beat best, we can stop.
        if a * hi < best:
            break
        for b in range(a, lo - 1, -1):
            prod = a * b
            # As b decreases, prod decreases; once <= best, no need to continue.
            if prod <= best:
                break
            s = str(prod)
            if s == s[::-1]:
                best = prod
                # For fixed a, this is the largest palindrome since b is descending.
                break
    return best


def solve(n: int) -> int:
    """
    Return the largest palindrome made from two n-digit factors.

    @formalsw: Py from project_euler_statements.p4 import naive
    @formalsw: Requires(n >= 0)
    @formalsw: Equiv(naive)
    """
    lo = 0 if n == 0 else 10 ** (n - 1)
    hi = 10 ** n - 1
    return largest_palindrome_product(lo, hi)


if __name__ == "__main__":
    assert solve(2) == 9009
    print(solve(3))
