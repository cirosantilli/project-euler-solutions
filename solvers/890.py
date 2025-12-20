#!/usr/bin/env python3
"""
Project Euler 890

We count binary partitions p(n): number of ways to write n as an unordered sum of powers of 2.
The classic recurrence implies:
    p(2m) = p(2m+1) = S(m)
where S(m) = sum_{i=0..m} p(i).

Let A(x) = sum_{m>=0} S(m) x^m.
Then:
    A(x) = P(x)/(1-x)
where P(x) = prod_{k>=0} 1/(1 - x^{2^k}) is the binary-partition generating function.
Using the identity 1/(1 - t) = prod_{j>=0} (1 + t^{2^j}), we obtain:
    A(x) = prod_{k>=0} (1 + x^{2^k})^{k+2}.

So the task reduces to extracting the coefficient [x^m] A(x) modulo MOD, with m = n//2.

We compute this coefficient with a carry-DP in base 2. Each DP step requires a convolution
with a binomial row, and we accelerate these convolutions by packing coefficient arrays
into a huge base (2^80) and using Python's big-integer multiplication.
"""

MOD = 1_000_000_007

# Packing base. Must exceed the maximum possible raw convolution coefficient.
# Max raw coefficient is <= L * (MOD-1)^2 where L <= ~2200 here, ~2.5e21 < 2^80.
BASE_BITS = 80
DIGIT_BYTES = BASE_BITS // 8
BASE_MASK = (1 << BASE_BITS) - 1


def _pack_digits(values):
    """Pack a list of nonnegative ints into one integer using base 2^BASE_BITS (little-endian)."""
    b = bytearray(DIGIT_BYTES * len(values))
    off = 0
    for v in values:
        b[off : off + DIGIT_BYTES] = v.to_bytes(DIGIT_BYTES, "little")
        off += DIGIT_BYTES
    return int.from_bytes(b, "little")


def _convolve_and_decimate(a, b, bit):
    """
    Compute c = a * b (ordinary convolution over integers),
    then return [ c[bit], c[bit+2], c[bit+4], ... ] reduced mod MOD.
    Uses base-2^80 packing + big integer multiplication to do convolution quickly.
    """
    la, lb = len(a), len(b)
    out_len = la + lb - 1

    ia = _pack_digits(a)
    ib = _pack_digits(b)
    prod = ia * ib

    # Convert product to fixed-length bytes so we can slice out digits.
    bs = prod.to_bytes(DIGIT_BYTES * out_len, "little")

    # We need indices t = bit + 2*i such that 0 <= t < out_len
    new_len = (out_len - bit + 1) // 2
    res = [0] * new_len
    for i in range(new_len):
        idx = bit + 2 * i
        start = idx * DIGIT_BYTES
        res[i] = int.from_bytes(bs[start : start + DIGIT_BYTES], "little") % MOD
    return res


def _prepare_factorials(nmax):
    """Precompute factorials and inverse factorials mod MOD up to nmax."""
    fact = [1] * (nmax + 1)
    invfact = [1] * (nmax + 1)
    for i in range(1, nmax + 1):
        fact[i] = (fact[i - 1] * i) % MOD
    invfact[nmax] = pow(fact[nmax], MOD - 2, MOD)
    for i in range(nmax, 0, -1):
        invfact[i - 1] = (invfact[i] * i) % MOD
    return fact, invfact


def coefficient_A(m):
    """
    Return [x^m] A(x) mod MOD where:
        A(x) = prod_{k>=0} (1 + x^{2^k})^{k+2}.
    Only k <= floor(log2(m)) can contribute.

    This coefficient equals S(m) = sum_{i=0..m} p(i).
    """
    if m < 0:
        return 0
    if m == 0:
        return 1

    L = m.bit_length()  # highest relevant k is L-1
    max_m = L + 2  # need binom(k+2, j) where k <= L-1
    fact, invfact = _prepare_factorials(max_m)

    dp = [1]  # dp[carry] at current bit position
    for k in range(L):
        bit = (m >> k) & 1
        top = k + 2  # (1 + x^{2^k})^{k+2}
        # Binomial row C(top, j), j=0..top
        row = [0] * (top + 1)
        ft = fact[top]
        for j in range(top + 1):
            row[j] = ft * invfact[j] % MOD * invfact[top - j] % MOD

        dp = _convolve_and_decimate(dp, row, bit)

    # After processing all bits of m, no further high terms can contribute,
    # so the final carry must be 0.
    return dp[0] % MOD


def p_binary_partitions(n):
    """
    p(n) = number of partitions of n into powers of two.
    Using p(2m)=p(2m+1)=S(m), we have:
        p(n) = S(n//2) = coefficient_A(n//2).
    """
    return coefficient_A(n // 2)


def main():
    # Test values from the problem statement
    assert p_binary_partitions(7) == 6

    n_test = pow(7, 7)
    assert p_binary_partitions(n_test) == 144548435

    # Actual required computation
    n = pow(7, 777)
    print(p_binary_partitions(n))


if __name__ == "__main__":
    main()
