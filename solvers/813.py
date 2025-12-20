#!/usr/bin/env python3
"""
Project Euler 813 - XOR-Powers

We define x ⊕ y as bitwise XOR and x ⊗ y as the "XOR-product" (carryless multiplication):
binary long multiplication where intermediate rows are XORed instead of added.

Let P(n) = 11^{⊗ n} = 11 ⊗ 11 ⊗ ... ⊗ 11 (n times).
Compute P(8^12 * 12^8) modulo 1_000_000_007.

No external libraries are used.
"""

MOD = 1_000_000_007


def xor_product(x: int, y: int) -> int:
    """Carryless (XOR) product of nonnegative integers x and y."""
    if x < 0 or y < 0:
        raise ValueError("xor_product expects nonnegative integers")
    res = 0
    shift = 0
    while y:
        if y & 1:
            res ^= x << shift
        y >>= 1
        shift += 1
    return res


def xor_pow(base: int, exp: int) -> int:
    """Exponentiation under xor_product (useful only for small exp in tests)."""
    if exp < 0:
        raise ValueError("exp must be nonnegative")
    result = 1  # multiplicative identity for ⊗
    while exp:
        if exp & 1:
            result = xor_product(result, base)
        base = xor_product(base, base)
        exp >>= 1
    return result


def degrees_for_power_of_11(exp: int) -> set[int]:
    """
    Return the set of degrees d such that x^d has coefficient 1 in
    (1 + x + x^3)^exp over GF(2).

    11 in binary is 1011_2, i.e. polynomial 1 + x + x^3.
    Over GF(2), multinomial coefficients are taken modulo 2.
    Using Lucas' theorem (p=2), only partitions of the 1-bits of exp contribute.

    Each 1-bit value v = 2^b of exp can be assigned to:
      - i (the constant term): contributes +0 to degree
      - j (the x term):        contributes +v to degree
      - k (the x^3 term):      contributes +3v to degree

    Coefficients are XOR/parity, so duplicate degrees cancel.
    """
    if exp < 0:
        raise ValueError("exp must be nonnegative")

    # Collect the values (powers of two) of set bits of exp.
    bit_values = []
    v = 1
    e = exp
    while e:
        if e & 1:
            bit_values.append(v)
        e >>= 1
        v <<= 1

    degs: set[int] = {0}
    for v in bit_values:
        new: set[int] = set()
        for d in degs:
            for add in (0, v, 3 * v):
                nd = d + add
                if nd in new:
                    new.remove(nd)  # XOR cancellation
                else:
                    new.add(nd)
        degs = new
    return degs


def solve() -> int:
    exp = pow(8, 12) * pow(12, 8)
    degs = degrees_for_power_of_11(exp)

    ans = 0
    for d in degs:
        ans = (ans + pow(2, d, MOD)) % MOD
    return ans


if __name__ == "__main__":
    # Tests from the problem statement:
    assert xor_product(11, 11) == 69
    assert xor_pow(11, 2) == 69  # P(2) = 69

    print(solve())
