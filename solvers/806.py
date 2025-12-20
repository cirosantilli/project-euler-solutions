#!/usr/bin/env python3
"""
Project Euler 806: Nim on Towers of Hanoi

We must compute f(n) modulo 1_000_000_007, where f(n) is the sum of indices i
in the unique shortest 3-peg Hanoi solution (positions indexed 0..2^n-1)
for which the Nim position (heap sizes = disk counts on the three pegs)
is losing for the first player.

Crucial symmetry:
Index i mirrors to (2^n-1-i) by swapping peg 1 and peg 3, which preserves XOR.
Thus losing indices come in pairs summing to 2^n-1, so

    f(n) = k * (2^n - 1) / 2   (mod M)

where k is the number of losing positions.
So we only need k, i.e. the total number of positions whose (a,b,c) satisfy:
    a+b+c = n  and  a xor b xor c = 0

We compute k via a generating function for the number of occurrences of each
(a,b,c) in the optimal Hanoi sequence, then sum coefficients over all XOR-zero
triples (a,b,c) with sum n.
"""

MOD = 1_000_000_007
INV2 = (MOD + 1) // 2


def mod_pow(a, e, mod=MOD):
    r = 1
    while e:
        if e & 1:
            r = (r * a) % mod
        a = (a * a) % mod
        e >>= 1
    return r


def precompute_factorials(nmax):
    """factorials, inv, invfactorials up to nmax (inclusive) modulo MOD."""
    fact = [1] * (nmax + 1)
    for i in range(1, nmax + 1):
        fact[i] = (fact[i - 1] * i) % MOD

    inv = [0] * (nmax + 1)
    inv[1] = 1
    for i in range(2, nmax + 1):
        inv[i] = MOD - (MOD // i) * inv[MOD % i] % MOD

    invfact = [1] * (nmax + 1)
    invfact[nmax] = mod_pow(fact[nmax], MOD - 2)
    for i in range(nmax, 0, -1):
        invfact[i - 1] = (invfact[i] * i) % MOD

    return fact, inv, invfact


def build_pow2(nmax):
    p = [1] * (nmax + 1)
    for i in range(1, nmax + 1):
        p[i] = (p[i - 1] * 2) % MOD
    return p


class CoefficientComputer:
    """
    Computes coefficients for:
        Fy(x,y,z) = ((1+y)(1+x+z-y)) / (1 - x^2 - y^2 - z^2 - 2xyz)

    For fixed (a,b,c), coefficient [x^a y^b z^c] Fy equals
    the number of times the Hanoi solution (with a+b+c disks)
    has exactly a disks on peg1, b on peg2, c on peg3.

    Denominator inversion:
      1 / (1 - (x^2 + y^2 + z^2 + 2xyz))
      = sum_{m>=0} (x^2 + y^2 + z^2 + 2xyz)^m

    Extracting [x^a y^b z^c] is done by choosing i times the term (2xyz)
    and the remainder as x^2,y^2,z^2 terms. This yields a multinomial sum.

    We accelerate the inner sum over i by using a multiplicative recurrence
    that steps i by 2 (parity constraint).
    """

    __slots__ = ("fact", "inv", "invfact", "pow2", "cache")

    def __init__(self, nmax):
        self.fact, self.inv, self.invfact = precompute_factorials(nmax)
        self.pow2 = build_pow2(nmax)
        self.cache = {}

    def _denom_coeff(self, a, b, c):
        """
        Coefficient of x^a y^b z^c in:
            1 / (1 - x^2 - y^2 - z^2 - 2xyz)
        """
        key = (a, b, c)
        if key in self.cache:
            return self.cache[key]

        if a < 0 or b < 0 or c < 0:
            self.cache[key] = 0
            return 0
        # parity constraint: a,b,c must share parity
        if ((a ^ b) & 1) or ((a ^ c) & 1):
            self.cache[key] = 0
            return 0

        minabc = a
        if b < minabc:
            minabc = b
        if c < minabc:
            minabc = c

        i = a & 1
        if i > minabc:
            self.cache[key] = 0
            return 0

        A = (a - i) // 2
        B = (b - i) // 2
        C = (c - i) // 2
        # m is the power in the series term
        m = (a + b + c - i) // 2

        fact = self.fact
        inv = self.inv
        invfact = self.invfact
        pow2 = self.pow2

        term = pow2[i]
        term = term * fact[m] % MOD
        term = term * invfact[i] % MOD
        term = term * invfact[A] % MOD
        term = term * invfact[B] % MOD
        term = term * invfact[C] % MOD

        ans = 0
        # Iterate i in steps of 2.
        while True:
            ans += term
            ans %= MOD

            i2 = i + 2
            if i2 > minabc:
                break

            # term_{i+2} / term_i = 4*A*B*C / (m*(i+1)*(i+2))
            ratio = (4 * A) % MOD
            ratio = (ratio * B) % MOD
            ratio = (ratio * C) % MOD
            ratio = (ratio * inv[m]) % MOD
            ratio = (ratio * inv[i + 1]) % MOD
            ratio = (ratio * inv[i + 2]) % MOD

            term = (term * ratio) % MOD

            # update state
            i = i2
            A -= 1
            B -= 1
            C -= 1
            m -= 1

        self.cache[key] = ans
        return ans

    def full_coeff(self, a, b, c):
        """
        Coefficient in Fy with numerator:
            (1+y)(1+x+z-y) = 1 + x + z + xy + yz - y^2
        """
        res = self._denom_coeff(a, b, c)
        res += self._denom_coeff(a - 1, b, c)  # +x
        res += self._denom_coeff(a, b, c - 1)  # +z
        res += self._denom_coeff(a - 1, b - 1, c)  # +xy
        res += self._denom_coeff(a, b - 1, c - 1)  # +yz
        res -= self._denom_coeff(a, b - 2, c)  # -y^2
        res %= MOD
        return res


def xor_zero_triples(n):
    """
    Enumerate all ordered triples (a,b,c) of nonnegative ints such that:
        a + b + c = n
        a xor b xor c = 0

    For even n, solutions can be built bitwise from set bits of n:
    Each set bit at position p>=1 corresponds to two of (a,b,c) having bit (p-1)=1.
    There are 3 choices per set bit => 3^{popcount(n)} solutions.
    """
    if n & 1:
        return []

    bits = []
    x = n
    p = 0
    while x:
        if x & 1:
            bits.append(p)
        x >>= 1
        p += 1

    # bit 0 must be unset for even n
    # if it is set, n is odd and we returned already.
    triples = [(0, 0, 0)]
    for p in bits:
        if p == 0:
            # can't happen for even n
            continue
        v = 1 << (p - 1)
        new_list = []
        for a, b, c in triples:
            new_list.append((a + v, b + v, c))
            new_list.append((a + v, b, c + v))
            new_list.append((a, b + v, c + v))
        triples = new_list
    return triples


def count_losing_positions(n, coeff_comp):
    """
    k = number of indices i in the optimal Hanoi solution where (a,b,c) is Nim-losing.
    k = sum over all xor-zero triples (a,b,c) with sum n of coefficient(a,b,c).
    """
    if n & 1:
        return 0
    total = 0
    for a, b, c in xor_zero_triples(n):
        total += coeff_comp.full_coeff(a, b, c)
        total %= MOD
    return total


def solve(n):
    """
    Compute f(n) modulo MOD.
    """
    if n & 1:
        return 0
    # n<=100000 for this problem; safe to precompute up to n.
    coeff_comp = CoefficientComputer(n + 5)
    k = count_losing_positions(n, coeff_comp)
    val = (mod_pow(2, n) - 1) % MOD
    return (k * val % MOD) * INV2 % MOD


def main():
    # Problem statement test values
    assert solve(4) == 30
    assert solve(10) == 67518

    n = 100000
    print(solve(n))


if __name__ == "__main__":
    main()
