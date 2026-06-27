#!/usr/bin/env python3
"""
A positive integer is 'balanced' if the length of its longest strictly
decreasing subsequence (of decimal digits) equals the length of its longest
non-decreasing subsequence.

Approach (RSK / Robinson-Schensted for words):
  For a word over the 10-symbol digit alphabet with RSK shape lambda:
     longest non-decreasing subsequence = lambda_1   (first row length)
     longest strictly decreasing subsequence      = l(lambda)  (number of rows)
  'balanced'  <=>  lambda_1 == l(lambda).
  Since rows <= 10 (only 10 distinct digits), a balanced shape fits in a
  10x10 box, so a balanced number has at most 100 digits: finite problem.

  #words over [10] of shape lambda = f^lambda * s_lambda(1^10), where
     f^lambda        = number of standard Young tableaux (hook length formula)
     s_lambda(1^10)  = number of semistandard YT with entries in [10]
                     = prod_cells (10 + col - row) / hook.
  Their product therefore divides by the hook product TWICE (hook^2).

  Leading-zero correction: a word 0u is not a valid integer. For 0u,
     LNDS(0u) = LNDS(u) + 1,   LDS(0u) = LDS(u),
  so 0u is balanced iff u's shape mu satisfies mu_1 + 1 == l(mu).
  Subtract those, plus the single word "0".
"""
from math import factorial

MODULUS = 1_000_000_007
ALPHABET_SIZE = 10
MAX_SIDE = 10                       # RSK shapes have at most 10 rows over a 10-symbol alphabet
MAX_DIGITS = MAX_SIDE * MAX_SIDE    # balanced numbers have at most 100 digits


def partitions(max_part: int, max_len: int):
    """Yield every non-empty partition with parts <= max_part and <= max_len parts."""
    current: list[int] = []

    def rec(limit: int, remaining_len: int):
        if current:
            yield tuple(current)
        if remaining_len == 0:
            return
        for part in range(limit, 0, -1):
            current.append(part)
            yield from rec(part, remaining_len - 1)
            current.pop()

    yield from rec(max_part, max_len)


def shape_weight(shape: tuple[int, ...]) -> int:
    """f^shape * s_shape(1^ALPHABET_SIZE) as an exact integer.

    = size! * prod(ALPHABET_SIZE + col - row) / (prod hook)^2
    """
    column_heights = [
        sum(1 for row_length in shape if row_length >= col)
        for col in range(1, shape[0] + 1)
    ]
    numerator = factorial(sum(shape))
    denominator = 1
    for row_index, row_length in enumerate(shape, start=1):
        for col_index in range(1, row_length + 1):
            hook = (row_length - col_index) + (column_heights[col_index - 1] - row_index) + 1
            numerator *= ALPHABET_SIZE + col_index - row_index
            denominator *= hook * hook          # hook enters twice (f^lambda and s_lambda)
    assert numerator % denominator == 0
    return numerator // denominator


def count_balanced_positive(max_digits: int = MAX_DIGITS) -> int:
    """Number of balanced positive integers with at most `max_digits` digits, mod MODULUS."""
    balanced_all = 0
    # leading-zero balanced words to subtract; the single word "0" is one of them
    balanced_leading_zero = 1 if max_digits >= 1 else 0

    for shape in partitions(MAX_SIDE, MAX_SIDE):
        size = sum(shape)
        width = shape[0]          # lambda_1
        height = len(shape)       # l(lambda)
        weight = shape_weight(shape)

        if size <= max_digits and width == height:
            balanced_all += weight
        if size + 1 <= max_digits and height == width + 1:
            balanced_leading_zero += weight

    return (balanced_all - balanced_leading_zero) % MODULUS


def main() -> None:
    assert count_balanced_positive(4) == 2274, "self-check below 10^4 failed"
    print(count_balanced_positive())


if __name__ == "__main__":
    main()
