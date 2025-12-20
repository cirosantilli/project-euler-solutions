#!/usr/bin/env python3
"""
Project Euler 792 - "Too Many Twos"

We work with the 2-adic valuation v2(x): the largest r such that 2**r divides x.
The problem defines:
  S(n) = sum_{k=1..n} (-2)^k * C(2k,k)
  u(n) = v2(3*S(n) + 4)
and asks for:
  U(N) = sum_{n=1..N} u(n^3)  with N = 10_000.

Key idea (used in many 2-adic treatments):
The infinite series 3*sum_{k>=1} (-2)^k*C(2k,k) + 4 converges to 0 in the 2-adics.
So, in the 2-adics,
  3*S(n) + 4 = -3 * sum_{k>n} (-2)^k*C(2k,k).
Since -3 is odd, it does not change v2, hence
  u(n) = v2( sum_{k>n} R(k) ) where R(k) = (-2)^k*C(2k,k).

We never compute large binomial coefficients.
Instead we use the exact ratio:
  R(k+1)/R(k) = -4 * (2k+1)/(k+1),
and perform 2-adic arithmetic modulo 2^P on the odd parts only.
"""

import sys
from math import comb


def v2_int(x: int) -> int:
    """2-adic valuation of a nonnegative integer; v2(0) treated as +infty."""
    if x == 0:
        return 10**18
    return (x & -x).bit_length() - 1


def inv_mod_2power_odd(a: int, P: int, mask: int) -> int:
    """
    Modular inverse of an odd a modulo 2^P, using Newton iteration.
    (a must be odd, so gcd(a, 2^P)=1.)
    """
    x = 1  # correct mod 2 for any odd a
    # Each iteration doubles the number of correct bits; a few steps suffice.
    for _ in range(P.bit_length() + 1):
        x = (x * (2 - a * x)) & mask
    return x


def u(n: int, *, P0: int = 256, max_m: int = 220) -> int:
    """
    Compute u(n) = v2(3*S(n)+4) using a short 2-adic tail sum.

    We sum the tail terms R(k) for k = n+1..n+m.
    The remainder tail (k >= n+m+1) is divisible by 2^(n+m+1), since v2(R(k)) >= k.
    Therefore, if the partial sum has valuation < n+m+1, it must equal u(n).

    Arithmetic is performed on odd parts modulo 2^P; powers of two are tracked separately.
    """
    if n < 0:
        raise ValueError("n must be nonnegative")

    for P in (P0, P0 * 2, P0 * 4, P0 * 8):
        mod = 1 << P
        mask = mod - 1

        inv_cache = {}

        def inv_odd(a: int) -> int:
            a &= mask
            res = inv_cache.get(a)
            if res is None:
                res = inv_mod_2power_odd(a, P, mask)
                inv_cache[a] = res
            return res

        # First term in the tail: k = n+1
        k = n + 1

        # v2(R(k)) = v2((-2)^k) + v2(C(2k,k)) = k + popcount(k)
        exp = k + k.bit_count()

        # We normalize the odd part of R(k) to 1.
        # This scales the whole tail by an odd constant, which does not change v2 of the sum.
        odd = 1

        min_exp = None
        scaled_sum = 0  # equals (partial_sum / 2^min_exp) mod 2^P

        for m in range(1, max_m + 1):
            # Add current term (2^exp * odd) into the running partial sum.
            if min_exp is None:
                min_exp = exp
                scaled_sum = odd & mask
            else:
                if exp < min_exp:
                    shift = min_exp - exp
                    scaled_sum = ((scaled_sum << shift) & mask) if shift < P else 0
                    min_exp = exp
                    scaled_sum = (scaled_sum + odd) & mask
                else:
                    shift = exp - min_exp
                    if shift < P:
                        scaled_sum = (scaled_sum + ((odd << shift) & mask)) & mask

            # If our reduced sum is 0 modulo 2^P, we need higher precision.
            if scaled_sum == 0:
                break

            v_partial = min_exp + v2_int(scaled_sum)

            # Remainder from k >= n+m+1 is divisible by 2^(n+m+1).
            if v_partial < n + m + 1:
                return v_partial

            # Advance to next term using the ratio:
            #   R(k+1)/R(k) = -4 * (2k+1)/(k+1)
            denom = k + 1
            t = v2_int(denom)
            denom_odd = denom >> t  # odd

            odd_mult = (-(2 * k + 1) * inv_odd(denom_odd)) & mask
            odd = (odd * odd_mult) & mask

            exp += 2 - t
            k = denom

            # Cheap consistency check (can be disabled if desired):
            if exp != k + k.bit_count():
                raise RuntimeError("internal mismatch in valuation tracking")

        # try higher P
    raise RuntimeError("failed to determine u(n) with available precision")


def U(N: int) -> int:
    """Compute U(N) = sum_{n=1..N} u(n^3)."""
    total = 0
    for n in range(1, N + 1):
        total += u(n * n * n)
    return total


def _S_naive(n: int) -> int:
    """Naive S(n) for small n (used only for problem-statement checks)."""
    s = 0
    for k in range(1, n + 1):
        s += (-2) ** k * comb(2 * k, k)
    return s


def _self_test() -> None:
    # Values stated in the problem statement:
    assert _S_naive(4) == 980
    assert 3 * _S_naive(4) + 4 == 2944
    assert u(4) == 7
    assert u(20) == 24
    assert U(5) == 241


def main() -> None:
    _self_test()
    N = 10_000
    if len(sys.argv) >= 2:
        N = int(sys.argv[1])
    print(U(N))


if __name__ == "__main__":
    main()
