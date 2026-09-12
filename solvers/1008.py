#!/usr/bin/env python

from __future__ import annotations

MODULUS = 1_000_000_007
TARGET = 10_000_000


def functional_inverse_x10(limit: int) -> int:
    p = MODULUS

    # B_r(n) = (n!)^2 e_r(1/1^2, ..., 1/n^2), for r = 0, ..., 9.
    # The variables are unrolled because this loop runs ten million times.
    b0 = 1
    b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = 0

    # D_n = (2n-1)!! and
    # A_n = (n!)^2 D_n * sum_{j=10}^n E_9(j-1)/(j(2j-1)).
    double_factorial = 1
    accumulator = 0

    for n in range(1, limit + 1):
        n2 = n * n % p
        odd = 2 * n - 1

        # Use b9 and D_{n-1} before advancing them to n.
        accumulator = (
            n2 * odd % p * accumulator
            + n * double_factorial % p * b9
        ) % p

        b9 = (n2 * b9 + b8) % p
        b8 = (n2 * b8 + b7) % p
        b7 = (n2 * b7 + b6) % p
        b6 = (n2 * b6 + b5) % p
        b5 = (n2 * b5 + b4) % p
        b4 = (n2 * b4 + b3) % p
        b3 = (n2 * b3 + b2) % p
        b2 = (n2 * b2 + b1) % p
        b1 = (n2 * b1 + b0) % p
        b0 = n2 * b0 % p

        double_factorial = double_factorial * odd % p

    scale = b0 * double_factorial % p
    interpolation_sum = accumulator * pow(scale, p - 2, p) % p

    # [x^10] P_{N+1} = (-1)^(N-9) B_9(N), while the degree-N
    # interpolation polynomial contributes -interpolation_sum.
    monic_part = b9 if (limit - 9) % 2 == 0 else -b9
    return (monic_part - interpolation_sum) % p


def main() -> None:
    assert functional_inverse_x10(9) == 1
    assert functional_inverse_x10(10) == 48808808
    print(functional_inverse_x10(TARGET))


if __name__ == "__main__":
    main()
