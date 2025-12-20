#!/usr/bin/env python3
"""
Project Euler 945: XOR-Equation C

We use x ⊕ y for bitwise XOR and define XOR-product ⊗ as carryless multiplication
over GF(2) (aka polynomial multiplication mod 2).

The equation:
(a⊗a) ⊕ (2⊗a⊗b) ⊕ (b⊗b) = (c⊗c)

For each valid pair (a,b), c exists uniquely iff a⊗b has NO coefficients
in even degrees (i.e. bits 0,2,4,... are all zero). Thus F(N) reduces to counting
pairs 0<=a<=b<=N satisfying that property.

No external libraries are used.
"""

EVEN_MASK = 0x5555555555555555  # bits 0,2,4,...


# -----------------------------
# XOR-product (carryless multiply)
# -----------------------------
def xor_product(x: int, y: int) -> int:
    """Carryless multiplication over GF(2) using shift-and-xor."""
    res = 0
    while y:
        lsb = y & -y
        res ^= x << (lsb.bit_length() - 1)
        y ^= lsb
    return res


def equation_holds(a: int, b: int, c: int) -> bool:
    """Check the original Euler equation."""
    left = xor_product(a, a) ^ xor_product(2, xor_product(a, b)) ^ xor_product(b, b)
    right = xor_product(c, c)
    return left == right


def cond_pair(a: int, b: int) -> bool:
    """Condition for existence of c: (a⊗b) has no even-position bits."""
    return (xor_product(a, b) & EVEN_MASK) == 0


# -----------------------------
# Brute force for tiny tests
# -----------------------------
def brute_F(N: int) -> int:
    cnt = 0
    for a in range(N + 1):
        for b in range(a, N + 1):
            if cond_pair(a, b):
                cnt += 1
    return cnt


# -----------------------------
# Bit compaction helpers
# (split an integer into even/odd bit polynomials in u=t^2)
# -----------------------------
def _build_compact_tables_16():
    """Precompute even/odd-bit compaction for 16-bit chunks."""
    size = 1 << 16
    even16 = [0] * size
    odd16 = [0] * size
    for x in range(size):
        e = 0
        o = 0
        for i in range(8):
            b = 1 << (2 * i)
            if x & b:
                e |= 1 << i
            if x & (b << 1):
                o |= 1 << i
        even16[x] = e
        odd16[x] = o
    return even16, odd16


EVEN16, ODD16 = _build_compact_tables_16()


def split_u(x: int):
    """
    Return (E,O) where:
      E is x's bits at even positions compacted,
      O is x's bits at odd positions compacted.
    """
    e = 0
    o = 0
    shift = 0
    while x:
        chunk = x & 0xFFFF
        e |= EVEN16[chunk] << shift
        o |= ODD16[chunk] << shift
        x >>= 16
        shift += 8
    return e, o


# -----------------------------
# Polynomial gcd over GF(2) (bit representation)
# -----------------------------
def gf2_mod(a: int, b: int) -> int:
    """Polynomial remainder a mod b over GF(2)."""
    db = b.bit_length() - 1
    while a and (a.bit_length() - 1) >= db:
        a ^= b << ((a.bit_length() - 1) - db)
    return a


def gf2_gcd(a: int, b: int) -> int:
    """Polynomial gcd over GF(2)."""
    while b:
        a, b = b, gf2_mod(a, b)
    return a


# -----------------------------
# Counting ordered solutions S(N)
# -----------------------------
def ordered_full(bits: int) -> int:
    """
    Ordered solution count for range [0, 2^bits - 1] (inclusive).
    Closed form derived from recurrence:
        S(bits) = (2^(bits+1)*(3*bits+4) + (-1)^bits) / 9
    """
    sign = -1 if (bits & 1) else 1
    num = (1 << (bits + 1)) * (3 * bits + 4) + sign
    return num // 9


def count_a_for_upper(k: int, y: int) -> int:
    """
    For b = 2^k + y, count how many a in [0, 2^k - 1] satisfy the condition.
    Uses a compact polynomial-gcd characterization.
    """
    m = k // 2
    Y0, Y1 = split_u(y)

    if (k & 1) == 0:
        # k even: k=2m
        # If Y1=0 -> A0 must be 0, A1 free => 2^m
        if Y1 == 0:
            return 1 << m
        # Count = 2^{deg(gcd(Y0⊕u^m, u*Y1))}
        P = Y0 ^ (1 << m)
        g = gf2_gcd(P, Y1 << 1)
        return 1 << (g.bit_length() - 1)
    else:
        # k odd: k=2m+1
        # If Y0=0 -> A1 must be 0, A0 free => 2^{m+1}
        if Y0 == 0:
            return 1 << (m + 1)
        # Count = 2^{deg(gcd(Y0, u*(u^m⊕Y1)))}
        uQ = ((1 << m) ^ Y1) << 1
        g = gf2_gcd(Y0, uQ)
        return 1 << (g.bit_length() - 1)


def ordered_S(N: int) -> int:
    """
    Ordered count of pairs (a,b) with 0<=a,b<=N satisfying the condition.
    """
    if N < 0:
        return 0
    if N == 0:
        return 1

    bits = N.bit_length()
    all_ones = (1 << bits) - 1
    if N == all_ones:
        return ordered_full(bits)

    k = bits - 1
    M = 1 << k
    r = N - M

    base = ordered_full(k)
    cross = 0
    for y in range(r + 1):
        cross += count_a_for_upper(k, y)

    return base + 2 * cross


def F(N: int) -> int:
    """
    Required F(N): number of solutions with 0<=a<=b<=N.

    The condition is symmetric in (a,b), and the only diagonal solution is (0,0),
    so:
        F(N) = (S(N) + 1) / 2
    """
    return (ordered_S(N) + 1) // 2


# -----------------------------
# Main
# -----------------------------
def main():
    # Statement examples / given values:
    assert xor_product(7, 3) == 9
    assert equation_holds(1, 2, 1)
    assert equation_holds(1, 8, 13)

    # Given test value:
    assert F(10) == 21
    # (Optional safety cross-check for small N)
    assert brute_F(10) == 21

    # Compute required value:
    print(F(10**7))


if __name__ == "__main__":
    main()
