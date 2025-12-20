#!/usr/bin/env python3
"""Project Euler 831: Triple Product

We need the first ten (most significant) base-7 digits of g(142857), where

  g(m) = sum_{j=0..m} sum_{i=0..j} (-1)^{j-i} C(m,j) C(j,i) C(j+5+6i, j+5).

No external libraries are used.
"""

from __future__ import annotations

from math import comb


DEG = 5  # we only ever need coefficients up to x^5


def poly_mul_trunc(a: list[int], b: list[int], deg: int = DEG) -> list[int]:
    """Multiply two polynomials and truncate to degree <= deg."""
    res = [0] * (deg + 1)
    for i, ai in enumerate(a):
        if ai == 0:
            continue
        for j, bj in enumerate(b):
            if bj == 0:
                continue
            k = i + j
            if k <= deg:
                res[k] += ai * bj
    return res


def poly_pow_trunc(base: list[int], exp: int, deg: int = DEG) -> list[int]:
    """Compute (base(x) ** exp) truncated to degree <= deg."""
    res = [0] * (deg + 1)
    res[0] = 1
    b = base[:]
    e = exp
    while e > 0:
        if e & 1:
            res = poly_mul_trunc(res, b, deg)
        e >>= 1
        if e:
            b = poly_mul_trunc(b, b, deg)
    return res


def coeff_c(m: int) -> int:
    """Return c_m where g(m) = 7^m * c_m.

    From the algebraic simplification:

      g(m) = [x^{m+5}] (1+x)^5 * ((1+x)^7 - 1)^m

    Write ((1+x)^7 - 1) = 7x * Q(x) where
      Q(x) = 1 + 3x + 5x^2 + 5x^3 + 3x^4 + x^5 + (1/7)x^6.

    To get [x^{m+5}] we only need [x^5] of (1+x)^5 * Q(x)^m, hence we can
    drop the x^6 term safely.
    """

    # Truncated Q(x) (the x^6 term can never contribute to x^5).
    Q = [1, 3, 5, 5, 3, 1]  # degrees 0..5

    Qm = poly_pow_trunc(Q, m, DEG)

    # (1+x)^5
    A = [comb(5, k) for k in range(DEG + 1)]

    AQm = poly_mul_trunc(A, Qm, DEG)
    return AQm[DEG]


def g_value(m: int) -> int:
    """Compute g(m) exactly (for small m)."""
    return (7**m) * coeff_c(m)


def to_base(n: int, base: int) -> str:
    """Convert a non-negative integer to a base-'base' string."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return "0"
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    out: list[str] = []
    while n:
        n, r = divmod(n, base)
        out.append(digits[r])
    out.reverse()
    return "".join(out)


def first_digits_base7_of_g(m: int, k: int = 10) -> str:
    """Return the first k base-7 digits of g(m)."""
    c = coeff_c(m)

    # g(m) = 7^m * c, so in base 7 it's base7(c) followed by m zeros.
    s = to_base(c, 7)

    if len(s) >= k:
        return s[:k]

    # If c has fewer than k digits, the leading digits include trailing zeros
    # coming from the 7^m factor.
    need = k - len(s)
    return s + ("0" * need)


def _self_test() -> None:
    # Given in the problem statement:
    assert g_value(10) == 127278262644918
    assert str(g_value(10))[:5] == "12727"


def main() -> None:
    _self_test()
    m = 142857
    print(first_digits_base7_of_g(m, 10))


if __name__ == "__main__":
    main()
