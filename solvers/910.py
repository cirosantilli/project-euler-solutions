#!/usr/bin/env python
"""
Project Euler 910: L-expressions II

The nested expression is evaluated in Z/(10^9)Z[x] modulo
prod_{k=1}^{40} (x + k).  Every polynomial is represented by its unique
degree-39 remainder in that quotient ring.
"""

DEGREE = 40
MOD = 10**9

DEFAULT_N = 12
DEFAULT_M = 345_678
DEFAULT_L = 9_012_345
DEFAULT_T = 678
DEFAULT_S = 90


def modulus_polynomial_fold() -> list[int]:
    """Return coefficients used to replace x^40 by a degree-39 polynomial."""
    coeffs = [1]
    for k in range(1, DEGREE + 1):
        nxt = [0] * (len(coeffs) + 1)
        for i, coeff in enumerate(coeffs):
            nxt[i] = (nxt[i] + coeff * k) % MOD
            nxt[i + 1] = (nxt[i + 1] + coeff) % MOD
        coeffs = nxt
    return [(-coeffs[i]) % MOD for i in range(DEGREE)]


FOLD = modulus_polynomial_fold()
ZERO = [0] * DEGREE
ONE = [1] + [0] * (DEGREE - 1)
X = [0, 1] + [0] * (DEGREE - 2)


def add(a: list[int], b: list[int]) -> list[int]:
    return [(x + y) % MOD for x, y in zip(a, b)]


def mul_x(poly: list[int]) -> list[int]:
    """Multiply by x and reduce the x^40 term immediately."""
    overflow = poly[-1]
    out = [0] + poly[:-1]
    if overflow:
        for i, coeff in enumerate(FOLD):
            out[i] = (out[i] + overflow * coeff) % MOD
    return out


def mul(a: list[int], b: list[int]) -> list[int]:
    """Multiply two quotient-ring elements."""
    out = [0] * DEGREE
    shifted = b[:]
    for coeff in a:
        if coeff:
            for i, val in enumerate(shifted):
                out[i] = (out[i] + coeff * val) % MOD
        shifted = mul_x(shifted)
    return out


def pow_poly(base: list[int], exponent: int) -> list[int]:
    result = ONE[:]
    power = base[:]
    while exponent:
        if exponent & 1:
            result = mul(power, result)
        exponent >>= 1
        if exponent:
            power = mul(power, power)
    return result


def compose(outer: list[int], inner: list[int]) -> list[int]:
    """Return outer(inner(x)) in the quotient ring."""
    result = ZERO[:]
    for coeff in reversed(outer):
        result = mul(result, inner)
        result[0] = (result[0] + coeff) % MOD
    return result


def iterate_composition(poly: list[int], count: int) -> list[int]:
    """Return poly composed with itself count times."""
    result = X[:]
    power = poly[:]
    while count:
        if count & 1:
            result = compose(power, result)
        count >>= 1
        if count:
            power = compose(power, power)
    return result


def d1(length: int) -> list[int]:
    base = pow_poly(X, length)
    return add(base, mul_x(base))


def d2(count: int, poly: list[int]) -> list[int]:
    return compose(iterate_composition(poly, count), mul_x(poly))


def expression_poly(nesting: int, middle_count: int, length: int) -> list[int]:
    base = d1(length)
    current = compose(base, d2(middle_count, base))
    for _ in range(nesting):
        current = d2(middle_count, current)
    return current


def evaluate(poly: list[int], x: int) -> int:
    result = 0
    for coeff in reversed(poly):
        result = (result * x + coeff) % MOD
    return result


def solve(
    nesting: int = DEFAULT_N,
    middle_count: int = DEFAULT_M,
    length: int = DEFAULT_L,
    x: int = DEFAULT_T,
    addend: int = DEFAULT_S,
) -> int:
    poly = expression_poly(nesting, middle_count, length)
    return (evaluate(poly, x) + addend) % MOD


def main() -> None:
    assert solve(0, 1, 1, 1, 0) == 42
    print(solve())


if __name__ == "__main__":
    main()
