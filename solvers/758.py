#!/usr/bin/env python3
"""
Project Euler 758: Buckets of Water

We need:
  Sum_{primes p<q<1000} P(2^{p^5}-1, 2^{q^5}-1) (mod 1_000_000_007)

Where P(a,b) is the minimal number of pourings to measure exactly 1 litre
using buckets (a, b, a+b) starting from (a, b, 0), with the rule that each
pour pours until the source is empty or the destination is full.

This program uses:
  - A continued-fraction characterization of P(a,b)
  - A Mersenne-number Euclidean step reduction to Euclid on exponents
  - Modular arithmetic for gigantic values
"""

from math import gcd

MOD = 1_000_000_007


# ----------------------------
# Helpers: primes under 1000
# ----------------------------
def primes_below(n: int) -> list[int]:
    """Return list of primes < n."""
    if n <= 2:
        return []
    sieve = bytearray(b"\x01") * n
    sieve[0:2] = b"\x00\x00"
    p = 2
    while p * p < n:
        if sieve[p]:
            step = p
            start = p * p
            sieve[start:n:step] = b"\x00" * (((n - 1 - start) // step) + 1)
        p += 1
    return [i for i in range(2, n) if sieve[i]]


# ------------------------------------------
# Continued fractions -> penultimate convergent
# ------------------------------------------
def penultimate_convergent_mod(
    cf_terms_mod: list[int], mod: int = MOD
) -> tuple[int, int]:
    """
    Given continued fraction terms [a0,a1,...,ak], return (p,q) for the
    penultimate convergent p/q (all computed mod 'mod').

    If there is only one convergent, returns that.
    """
    p_m2, p_m1 = 0, 1
    q_m2, q_m1 = 1, 0
    convs: list[tuple[int, int]] = []
    for a in cf_terms_mod:
        p = (a * p_m1 + p_m2) % mod
        q = (a * q_m1 + q_m2) % mod
        convs.append((p, q))
        p_m2, p_m1 = p_m1, p
        q_m2, q_m1 = q_m1, q

    if not convs:
        raise ValueError("continued fraction must have at least one term")
    return convs[0] if len(convs) == 1 else convs[-2]


def P_from_contfrac_penultimate_mod(p: int, q: int, mod: int = MOD) -> int:
    """P(a,b) ≡ 2*(p+q)-2 (mod mod), where p/q is the penultimate convergent of b/a."""
    return (2 * ((p + q) % mod) - 2) % mod


# ----------------------------------------------------
# Exact P(a,b) for small integers (for provided asserts)
# ----------------------------------------------------
def P_int(a: int, b: int) -> int:
    """
    Compute P(a,b) exactly for ordinary integers a<=b with gcd(a,b)=1,
    using the continued fraction of b/a.

    This is only used for the given test values in the statement.
    """
    if a <= 0 or b <= 0 or a > b:
        raise ValueError("require positive integers with a<=b")
    if gcd(a, b) != 1:
        raise ValueError("require gcd(a,b)=1")

    # Continued fraction of b/a by Euclid
    terms: list[int] = []
    num, den = b, a
    while den:
        q = num // den
        terms.append(q)
        num, den = den, num - q * den

    # Convergents
    p_m2, p_m1 = 0, 1
    q_m2, q_m1 = 1, 0
    convs: list[tuple[int, int]] = []
    for t in terms:
        p = t * p_m1 + p_m2
        q = t * q_m1 + q_m2
        convs.append((p, q))
        p_m2, p_m1 = p_m1, p
        q_m2, q_m1 = q_m1, q

    pen_p, pen_q = convs[0] if len(convs) == 1 else convs[-2]
    return 2 * (pen_p + pen_q) - 2


# -------------------------------------------------------
# Mersenne reduction: continued fraction for (2^E - 1)/(2^F - 1)
# -------------------------------------------------------
_inv_cache: dict[int, int] = {}


def _inv_ratio_minus1(ratio: int) -> int:
    """Return modular inverse of (ratio-1) mod MOD, caching by ratio."""
    inv = _inv_cache.get(ratio)
    if inv is None:
        inv = pow((ratio - 1) % MOD, MOD - 2, MOD)
        _inv_cache[ratio] = inv
    return inv


def _geom_sum_ratio(ratio: int, m: int) -> int:
    """
    Return S = 1 + ratio + ... + ratio^(m-1) (mod MOD), for m>=1.
    Uses closed form when ratio!=1, otherwise S=m.
    """
    if m <= 0:
        return 0
    ratio %= MOD
    if ratio == 1:
        return m % MOD
    inv = _inv_ratio_minus1(ratio)
    return ((pow(ratio, m, MOD) - 1) % MOD) * inv % MOD


def cf_terms_mersenne_exponents(e_small: int, e_large: int) -> list[int]:
    """
    Continued fraction terms for:
        (2^{e_large} - 1) / (2^{e_small} - 1)

    without constructing the huge integers, returning each term modulo MOD.

    Uses the identity (for e_large = m*e_small + r):
        2^{e_large}-1 = Q*(2^{e_small}-1) + (2^{r}-1),
    where Q = sum_{j=0}^{m-1} 2^{r + j*e_small}.
    """
    if not (1 <= e_small < e_large):
        raise ValueError("require 1 <= e_small < e_large")

    terms: list[int] = []
    hi, lo = e_large, e_small
    while True:
        m, r = divmod(hi, lo)

        ratio = pow(2, lo, MOD)  # 2^{lo} (mod MOD)
        shift = pow(2, r, MOD)  # 2^{r}  (mod MOD)
        series = _geom_sum_ratio(ratio, m)  # sum_{j=0}^{m-1} ratio^j (mod MOD)
        q_mod = (shift * series) % MOD  # shift * series (mod MOD)
        terms.append(q_mod)

        if r == 0:
            break
        hi, lo = lo, r

    return terms


def P_mersenne_exponents_mod(ea: int, eb: int) -> int:
    """
    Return P(2^{ea}-1, 2^{eb}-1) mod MOD, assuming 1<=ea<eb and gcd(ea,eb)=1.
    """
    terms_mod = cf_terms_mersenne_exponents(ea, eb)
    pen_p, pen_q = penultimate_convergent_mod(terms_mod, MOD)
    return P_from_contfrac_penultimate_mod(pen_p, pen_q, MOD)


# ----------------------------
# Main solve
# ----------------------------
def solve() -> int:
    primes = primes_below(1000)
    exps = [p**5 for p in primes]

    total = 0
    for i in range(len(primes)):
        ea = exps[i]
        for j in range(i + 1, len(primes)):
            eb = exps[j]
            # gcd(ea, eb) == 1 for distinct primes p,q (since ea=p^5, eb=q^5)
            total = (total + P_mersenne_exponents_mod(ea, eb)) % MOD
    return total


def _self_test() -> None:
    # Asserts from the problem statement:
    assert P_int(3, 5) == 4
    assert P_int(7, 31) == 20
    assert P_int(1234, 4321) == 2780


if __name__ == "__main__":
    _self_test()
    print(solve())
