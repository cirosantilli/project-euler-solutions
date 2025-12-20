#!/usr/bin/env python3
"""
Project Euler 866: Tidying Up B

Compute the expected value (an integer) of the product described in the problem,
for N=100, modulo 987654319.

No third-party libraries are used.
"""

MOD = 987654319


def expected_product_value(n: int, mod: int = MOD) -> int:
    """
    Returns E(n) mod `mod`, where E(n) is the expected value of the product.
    Uses the recurrence:
        E(0) = 1
        E(n) = (2n - 1) * sum_{k=0..n-1} E(k) * E(n-1-k)
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return 1 % mod

    E = [0] * (n + 1)
    E[0] = 1 % mod

    for length in range(1, n + 1):
        s = 0
        # Convolution of E[0..length-1] with itself reversed
        for k in range(length):
            s = (s + E[k] * E[length - 1 - k]) % mod
        E[length] = ((2 * length - 1) * s) % mod

    return E[n]


def _tests() -> None:
    # Test value given in the problem statement:
    assert expected_product_value(4, MOD) == 994

    # Small sanity checks (derived from the recurrence / easy cases):
    assert expected_product_value(0, MOD) == 1
    assert expected_product_value(1, MOD) == 1
    assert expected_product_value(2, MOD) == 6


def main() -> None:
    _tests()
    print(expected_product_value(100, MOD))


if __name__ == "__main__":
    main()
