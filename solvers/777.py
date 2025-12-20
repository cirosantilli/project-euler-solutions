#!/usr/bin/env python3
"""
Project Euler 777 — Lissajous Curves

Compute s(10^6) and print it in scientific notation rounded to 10 significant digits.

No external libraries are used.
"""

from __future__ import annotations


def d_num4(a: int, b: int) -> int:
    """
    Return 4*d(a,b) as an integer.

    Empirically and analytically (see README), for coprime a,b:
      - if 10 ∤ ab : d(a,b) = (4ab - 3a - 3b)/2
      - if 10 | ab : d(a,b) = (2ab - 3a - 3b + 4)/4

    This function returns the numerator over 4 in both cases.
    """
    ab = a * b
    if ab % 10 == 0:
        return 2 * ab - 3 * a - 3 * b + 4
    else:
        return 8 * ab - 6 * a - 6 * b


def _sum_first(n: int) -> int:
    """1 + 2 + ... + n"""
    return n * (n + 1) // 2


def mobius_sieve(n: int) -> list[int]:
    """
    Linear sieve for Möbius μ(k) for k=0..n (μ(0) unused).
    """
    mu = [0] * (n + 1)
    mu[1] = 1
    primes: list[int] = []
    is_comp = bytearray(n + 1)

    for i in range(2, n + 1):
        if not is_comp[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            ip = i * p
            if ip > n:
                break
            is_comp[ip] = 1
            if i % p == 0:
                mu[ip] = 0
                break
            mu[ip] = -mu[i]
    return mu


def _special_inner(n: int, need2: bool, need5: bool) -> tuple[int, int, int]:
    """
    Over x,y in [1..n], compute:
      cnt   = #pairs with (if need2 then 2|xy) and (if need5 then 5|xy)
      sumX  = Σ x over those pairs
      sumXY = Σ x*y over those pairs

    The conditions depend only on whether x or y lie in small residue classes,
    so we can use inclusion–exclusion with counts/sums of:
      odd numbers, numbers not divisible by 5, and numbers that are both odd and not divisible by 5.
    """
    total_cnt = n * n
    S = _sum_first(n)
    total_sumX = S * n
    total_sumXY = S * S

    if not need2 and not need5:
        return total_cnt, total_sumX, total_sumXY

    # Precompute counts and sums for needed complements
    ce = n // 2
    se = 2 * _sum_first(ce)  # sum of even numbers
    co = n - ce
    so = S - se  # sum of odd numbers

    c5 = n // 5
    s5 = 5 * _sum_first(c5)  # sum of multiples of 5
    cnot5 = n - c5
    snot5 = S - s5  # sum of non-multiples of 5

    c10 = n // 10
    s10 = 10 * _sum_first(c10)  # sum of multiples of 10

    # odd and not multiple of 5 = complement of (even OR multiple of 5)
    codnot5 = n - ce - c5 + c10
    sodnot5 = S - se - s5 + s10

    if need2 and not need5:
        # at least one even => exclude (odd, odd)
        cnt = total_cnt - co * co
        sumX = total_sumX - so * co
        sumXY = total_sumXY - so * so
        return cnt, sumX, sumXY

    if need5 and not need2:
        # at least one multiple of 5 => exclude (not5, not5)
        cnt = total_cnt - cnot5 * cnot5
        sumX = total_sumX - snot5 * cnot5
        sumXY = total_sumXY - snot5 * snot5
        return cnt, sumX, sumXY

    # need both:
    # exclude pairs failing even (odd,odd) and failing 5 (not5,not5), add back failing both (odd_not5,odd_not5)
    cnt = total_cnt - co * co - cnot5 * cnot5 + codnot5 * codnot5
    sumX = total_sumX - so * co - snot5 * cnot5 + sodnot5 * codnot5
    sumXY = total_sumXY - so * so - snot5 * snot5 + sodnot5 * sodnot5
    return cnt, sumX, sumXY


def s_num4(m: int) -> int:
    """
    Return 4*s(m) as an integer, where:
      s(m) = Σ d(a,b) over coprime integers 2<=a,b<=m.

    The computation uses Möbius inversion to handle gcd(a,b)=1 in O(m).
    """
    mu = mobius_sieve(m)

    # Sums over coprime pairs in [1..m]^2:
    A1 = 0  # Σ ab
    B1 = 0  # Σ a
    # Special subset (10|ab):
    Csp1 = 0  # count
    Bsp1 = 0  # Σ a
    Asp1 = 0  # Σ ab

    for d in range(1, m + 1):
        md = mu[d]
        if md == 0:
            continue
        n = m // d
        S = _sum_first(n)

        nn = n * n
        SS = S * S
        dd = d * d

        # all coprime (via Möbius)
        A1 += md * dd * SS
        B1 += md * d * S * n

        # special subset 10|ab depends on whether d already supplies factors 2 and/or 5
        need2 = (d & 1) == 1  # d is odd  => x*y must supply a factor 2
        need5 = (d % 5) != 0  # d not mult 5 => x*y must supply a factor 5
        cnt, sumX, sumXY = _special_inner(n, need2, need5)

        Csp1 += md * cnt
        Bsp1 += md * d * sumX
        Asp1 += md * dd * sumXY

    # Adjust from [1..m] to [2..m]
    sum1 = _sum_first(m)

    # For A1,B1 remove rows/cols where a=1 or b=1 (gcd always 1 there), add back (1,1)
    A = A1 - 2 * sum1 + 1
    B = B1 - m - sum1 + 1

    # For special subset: remove pairs where a=1 or b=1 and 10|ab, i.e. the other is a multiple of 10
    c10 = m // 10
    sum10 = 10 * _sum_first(c10)  # 10 + 20 + ... + 10*c10

    Csp = Csp1 - 2 * c10
    Bsp = Bsp1 - c10 * 1 - sum10
    Asp = Asp1 - sum10 - sum10

    # Generic contribution for all coprime pairs:
    # d_generic = (4ab - 3a - 3b)/2  => 4*d_generic summed is (8A - 12B)
    # Correction on special pairs where 10|ab:
    # 4*(d_special - d_generic) = -6ab + 3a + 3b + 4
    # Summed over special pairs: -6Asp + 6Bsp + 4Csp (since Σa = Σb by symmetry).
    num4 = (8 * A - 12 * B) + (-6 * Asp + 6 * Bsp + 4 * Csp)
    return num4


def _round_div(num: int, den: int) -> int:
    """Round num/den to nearest integer, ties rounded up (num,den>0)."""
    return (2 * num + den) // (2 * den)


def format_scientific(num: int, den: int = 1, sig: int = 10) -> str:
    """
    Format the rational num/den in scientific notation with `sig` significant digits.

    Output matches Project Euler style, e.g. "2.425650500e7" for sig=10.
    """
    if num == 0:
        return "0." + "0" * (sig - 1) + "e0"
    if den <= 0:
        raise ValueError("den must be positive")

    # Determine exponent e such that 1 <= (num/den) / 10^e < 10
    ip = num // den
    if ip > 0:
        e = len(str(ip)) - 1
    else:
        # value < 1: find smallest k so that num*10^k >= den
        k = 0
        t = num
        while t < den:
            t *= 10
            k += 1
        e = -k

    # mantissa_scaled = round( (num/den) * 10^{(sig-1)-e} )
    power = (sig - 1) - e
    if power >= 0:
        scaled_num = num * (10**power)
        scaled_den = den
    else:
        scaled_num = num
        scaled_den = den * (10 ** (-power))

    mant_scaled = _round_div(scaled_num, scaled_den)

    # Handle rounding overflow (e.g. 9.999... -> 10.000...)
    limit = 10**sig
    if mant_scaled >= limit:
        mant_scaled //= 10
        e += 1

    s = str(mant_scaled).zfill(sig)
    return f"{s[0]}.{s[1:]}e{e}"


def _self_test() -> None:
    # d(a,b) examples from the statement
    assert d_num4(2, 5) == 3  # 0.75 * 4
    assert d_num4(2, 3) == 18  # 4.5  * 4
    assert d_num4(7, 4) == 158  # 39.5 * 4
    assert d_num4(7, 5) == 208  # 52   * 4
    assert d_num4(10, 7) == 93  # 23.25* 4

    # s(m) examples from the statement
    assert s_num4(10) == 6410  # 1602.5 * 4
    assert s_num4(100) == 97026020  # 24256505 * 4


def main() -> None:
    _self_test()
    m = 10**6
    num4 = s_num4(m)
    print(format_scientific(num4, 4, sig=10))


if __name__ == "__main__":
    main()
