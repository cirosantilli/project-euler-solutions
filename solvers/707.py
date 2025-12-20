#!/usr/bin/env python3
"""
Project Euler 707 - Lights Out

No external libraries are used.

The key fact: the set of solvable starting states forms the image of the move matrix over GF(2),
so the number of solvable states is 2^rank (over GF(2)).
"""

MOD = 1_000_000_007
PHI = MOD - 1  # for base-2 exponent reduction mod MOD


# -----------------------------
# Polynomials over GF(2) as bitmasks
# -----------------------------
# A polynomial p(x) = sum p_i x^i is stored as an int with bit i = p_i.


def poly_deg(p: int) -> int:
    """Degree of polynomial (assumes p != 0)."""
    return p.bit_length() - 1


def poly_rem(a: int, b: int) -> int:
    """Remainder of a(x) divided by b(x) over GF(2). b must be nonzero."""
    db = poly_deg(b)
    while a and poly_deg(a) >= db:
        a ^= b << (poly_deg(a) - db)
    return a


def poly_gcd(a: int, b: int) -> int:
    """GCD of polynomials over GF(2)."""
    while b:
        a, b = b, poly_rem(a, b)
    return a


def poly_mod_f(a: int, f: int, deg_f: int) -> int:
    """Reduce a(x) modulo a monic polynomial f(x) of degree deg_f."""
    while a and poly_deg(a) >= deg_f:
        a ^= f << (poly_deg(a) - deg_f)
    return a


def poly_square_mod(a: int, f: int, deg_f: int) -> int:
    """
    Square a(x) over GF(2), then reduce mod f.

    In characteristic 2: (sum a_i x^i)^2 = sum a_i x^(2i),
    so squaring is just doubling exponents of set bits.
    """
    res = 0
    bits = a
    while bits:
        lsb = bits & -bits
        i = lsb.bit_length() - 1
        res |= 1 << (2 * i)
        bits ^= lsb
    return poly_mod_f(res, f, deg_f)


def poly_mul_x_mod(a: int, f: int, deg_f: int) -> int:
    """Compute x * a(x) mod f."""
    return poly_mod_f(a << 1, f, deg_f)


# -----------------------------
# Characteristic polynomial of L_w over GF(2)
# -----------------------------
# L_w is the w×w tridiagonal matrix with ones on the main diagonal and the two adjacent diagonals.
# Its characteristic polynomial is det(x I - L_w).
#
# For such tridiagonal matrices: D_0=1, D_1=(x+1), D_n=(x+1)D_{n-1}+D_{n-2} over GF(2).


def char_poly_L(w: int) -> int:
    """Return det(x I - L_w) as a GF(2) polynomial bitmask (degree w)."""
    if w < 0:
        raise ValueError("w must be nonnegative")
    if w == 0:
        return 1
    x_plus_1 = 0b11  # x + 1
    if w == 1:
        return x_plus_1
    d0, d1 = 1, x_plus_1
    for _ in range(2, w + 1):
        # (x+1)*d1 + d0 = (x*d1 + d1) + d0
        d2 = (d1 << 1) ^ d1 ^ d0
        d0, d1 = d1, d2
    return d1


# -----------------------------
# Fibonacci polynomials mod f in characteristic 2
# -----------------------------
# Define F_0=0, F_1=1, and F_{n+1} = x F_n + F_{n-1}.
#
# In characteristic 2 we get very simple fast-doubling:
#   F_{2k}   = x * (F_k)^2
#   F_{2k+1} = (F_k)^2 + (F_{k+1})^2
#
# All operations are performed in GF(2)[x] / (f(x)).


def fib_poly_mod(n: int, f: int, deg_f: int) -> tuple[int, int]:
    """
    Return (F_n mod f, F_{n+1} mod f) for Fibonacci polynomials over GF(2).
    Uses recursion depth O(log n).
    """
    if n == 0:
        return 0, 1

    a, b = fib_poly_mod(n >> 1, f, deg_f)  # (F_k, F_{k+1})
    a2 = poly_square_mod(a, f, deg_f)
    b2 = poly_square_mod(b, f, deg_f)

    c = poly_mul_x_mod(a2, f, deg_f)  # F_{2k}
    d = a2 ^ b2  # F_{2k+1}

    if (n & 1) == 0:
        return c, d
    # n = 2k+1: (F_{2k+1}, F_{2k+2})
    e = poly_mul_x_mod(d, f, deg_f) ^ c  # x*F_{2k+1} + F_{2k}
    return d, e


# -----------------------------
# Main solver for fixed width
# -----------------------------


class LightsOutWidthSolver:
    """
    For a fixed width w, compute:
      F(w, h) = number of solvable states on a w×h grid.
    """

    def __init__(self, w: int):
        if w <= 0:
            raise ValueError("w must be positive")
        self.w = w
        self.f = char_poly_L(w)  # characteristic polynomial of L_w over GF(2)
        self.deg_f = w

    def nullity(self, h: int) -> int:
        """
        Nullity (dimension of kernel) of the full (w*h) move matrix over GF(2).

        It equals deg(gcd(char_poly(L_w), F_{h+1}(x))) where F_n is the Fibonacci polynomial.
        """
        if h <= 0:
            raise ValueError("h must be positive")
        r, _ = fib_poly_mod(h + 1, self.f, self.deg_f)  # F_{h+1}(x) mod f
        g = poly_gcd(self.f, r)
        return poly_deg(g)  # degree of gcd

    def F(self, h: int, mod: int | None = MOD) -> int:
        """
        Number of solvable states F(w,h).

        If mod is None, returns the exact integer (only suitable for small w*h).
        Otherwise returns F(w,h) mod mod.
        """
        k = self.nullity(h)
        exp = self.w * h - k  # rank = w*h - nullity

        if mod is None:
            return 1 << exp
        # For MOD prime and base 2, reduce exponent mod (MOD-1).
        return pow(2, exp % (mod - 1), mod)


def fib_sequence(n: int) -> list[int]:
    """Return [f_0..f_n] with f_1=f_2=1 as in the statement (f_0 is set to 0)."""
    if n < 0:
        raise ValueError("n must be nonnegative")
    f = [0] * (n + 1)
    if n >= 1:
        f[1] = 1
    if n >= 2:
        f[2] = 1
    for i in range(3, n + 1):
        f[i] = f[i - 1] + f[i - 2]
    return f


def S(w: int, n: int, mod: int = MOD) -> int:
    """Compute S(w,n) = sum_{k=1..n} F(w, f_k) (mod mod)."""
    solver = LightsOutWidthSolver(w)
    fibs = fib_sequence(n)
    total = 0
    for k in range(1, n + 1):
        total = (total + solver.F(fibs[k], mod=mod)) % mod
    return total


def _run_asserts() -> None:
    # Given F(w,h) values
    assert LightsOutWidthSolver(1).F(2, mod=None) == 2
    assert LightsOutWidthSolver(3).F(3, mod=None) == 512
    assert LightsOutWidthSolver(4).F(4, mod=None) == 4096
    assert LightsOutWidthSolver(7).F(11, mod=MOD) == 270016253

    # Given S(w,n) values
    assert S(3, 3, mod=MOD) == 32
    assert S(4, 5, mod=MOD) == 1052960
    assert S(5, 7, mod=MOD) == 346547294


def solve() -> int:
    return S(199, 199, mod=MOD)


if __name__ == "__main__":
    _run_asserts()
    print(solve())
