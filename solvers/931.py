#!/usr/bin/env python3
"""Project Euler 931: Totient Graph

Compute T(10^12) modulo 715827883.

No external libraries are used.
"""

from __future__ import annotations

from array import array
from math import isqrt


MOD = 715_827_883
TARGET = 10**12


def sieve_primes(limit: int) -> list[int]:
    """Return a list of all primes <= limit using an odd-only sieve."""
    if limit < 2:
        return []
    if limit == 2:
        return [2]

    # odd numbers: index i represents (2*i + 1)
    size = (limit // 2) + 1
    is_comp = bytearray(size)
    primes = [2]

    r = isqrt(limit)
    for i in range(1, (r // 2) + 1):
        if not is_comp[i]:
            p = 2 * i + 1
            start = (p * p) // 2
            step = p
            is_comp[start::step] = b"\x01" * (((size - 1 - start) // step) + 1)

    for i in range(1, size):
        if not is_comp[i]:
            primes.append(2 * i + 1)
    return primes


def factorize(n: int, primes: list[int]) -> list[tuple[int, int]]:
    """Prime factorization using a provided prime list up to sqrt(n)."""
    out: list[tuple[int, int]] = []
    tmp = n
    for p in primes:
        if p * p > tmp:
            break
        if tmp % p == 0:
            e = 0
            while tmp % p == 0:
                tmp //= p
                e += 1
            out.append((p, e))
    if tmp > 1:
        out.append((tmp, 1))
    return out


def t_of_n(n: int, primes: list[int]) -> int:
    """Compute t(n) exactly from the closed form.

    If n = \prod p_i^{e_i}, then:
      t(n) = \sum_{p^e || n} (n/p^e) * ( (p-1)p^{e-1} - 1 )
    """
    fac = factorize(n, primes)
    total = 0
    for p, e in fac:
        pe = p**e
        total += (n // pe) * ((p - 1) * (p ** (e - 1)) - 1)
    return total


def T_naive(N: int) -> int:
    """Naive T(N) for small N, used only for asserts."""
    if N <= 1:
        return 0
    primes = sieve_primes(isqrt(N) + 10)
    s = 0
    for n in range(2, N + 1):
        s += t_of_n(n, primes)
    return s


def tri(x: int, mod: int) -> int:
    """F(x) = x(x+1)/2 modulo mod."""
    return (x * (x + 1) // 2) % mod


def min25_pi_and_prime_sum_mod(n: int, mod: int):
    """Compute pi(x) and sum_{p<=x} p (mod) for x in {1..sqrt(n)} U {n//i: 1<=i<=sqrt(n)}.

    This is the classic 'divisor-splitting' prime-sum/count sieve:
      g(x) starts as count of integers in [2..x]
      h(x) starts as sum of integers in [2..x]
    Then for each prime p, remove numbers whose smallest prime factor is p.

    Returns (v, primes, vals, g_small, h_small, g_large, h_large) where:
      v = floor(sqrt(n))
      vals[i] = n//i (1<=i<=v)
      g_small[x], h_small[x] for x<=v
      g_large[i], h_large[i] correspond to x = n//i

    After processing, g_small[x] == pi(x), g_large[i] == pi(n//i),
    and h_* hold prime sums modulo mod.
    """
    v = isqrt(n)
    primes = sieve_primes(v)

    # 1-indexed arrays for the large side: index i represents x = n//i
    vals = array("Q", [0]) * (v + 1)
    g_large = array("Q", [0]) * (v + 1)
    h_large = array("Q", [0]) * (v + 1)

    # 0..v arrays for the small side: index x represents x itself
    g_small = array("Q", [0]) * (v + 1)
    h_small = array("Q", [0]) * (v + 1)

    # Initialize
    for x in range(0, v + 1):
        if x >= 2:
            g_small[x] = x - 1
            h_small[x] = (x * (x + 1) // 2 - 1) % mod
        else:
            g_small[x] = 0
            h_small[x] = 0

    for i in range(1, v + 1):
        x = n // i
        vals[i] = x
        if x >= 2:
            g_large[i] = x - 1
            h_large[i] = (x * (x + 1) // 2 - 1) % mod
        else:
            g_large[i] = 0
            h_large[i] = 0

    def get_g_h(x: int) -> tuple[int, int]:
        if x <= v:
            return int(g_small[x]), int(h_small[x])
        # x is guaranteed to be exactly n//k for some 1<=k<=v when x>v
        k = n // x
        return int(g_large[k]), int(h_large[k])

    # Process primes
    for p in primes:
        p2 = p * p
        if p2 > n:
            break

        g_p1 = int(g_small[p - 1])
        h_p1 = int(h_small[p - 1])

        # Update large values first, in increasing i.
        i_max = n // p2
        if i_max > v:
            i_max = v

        for i in range(1, i_max + 1):
            y = int(vals[i] // p)  # equals floor(n / (i*p))
            g_y, h_y = get_g_h(y)

            g_large[i] -= g_y - g_p1

            diff = h_y - h_p1
            if diff < 0:
                diff += mod
            h_large[i] = (int(h_large[i]) - p * diff) % mod

        # Update small values descending so y=x//p hasn't been touched for this p.
        for x in range(v, p2 - 1, -1):
            y = x // p
            g_small[x] -= int(g_small[y]) - g_p1

            diff = int(h_small[y]) - h_p1
            if diff < 0:
                diff += mod
            h_small[x] = (int(h_small[x]) - p * diff) % mod

    return v, primes, vals, g_small, h_small, g_large, h_large


def compute_T_mod(N: int, mod: int) -> int:
    """Compute T(N) mod mod.

    Closed form for a single n:
      t(n) = sum_{p^e || n} (n/p^e) * (phi(p^e) - 1)

    Swapping sums yields per-(p,e) contributions:
      A(p,e) = (p-1)p^{e-1} - 1
      X_e = floor(N / p^e)
      contribution = A(p,e) * ( F(X_e) - p*F(X_{e+1}) )
      where F(x) = x(x+1)/2.

    For primes p > sqrt(N), only e=1 occurs and X_2=0; we group by q=floor(N/p).
    """
    if N <= 1:
        return 0

    sqrtN = isqrt(N)

    # Precompute pi(x) and sum_{p<=x} p (mod) for the required x values.
    v, primes, _vals, g_small, h_small, g_large, h_large = min25_pi_and_prime_sum_mod(
        N, mod
    )

    def pi(x: int) -> int:
        if x <= v:
            return int(g_small[x])
        return int(g_large[N // x])

    def psum(x: int) -> int:
        if x <= v:
            return int(h_small[x])
        return int(h_large[N // x])

    total = 0

    # Part 1: primes p <= sqrt(N), all exponents e>=1.
    for p in primes:
        # e=1
        x1 = N // p
        x2 = N // (p * p)
        f = (tri(x1, mod) - (p % mod) * tri(x2, mod)) % mod
        total = (total + ((p - 2) % mod) * f) % mod

        # e>=2
        pe = p * p
        p_pow = p  # p^{e-1} for current e, starts at e=2
        while pe <= N:
            xe = N // pe
            xnext = N // (pe * p)
            A = (p - 1) * p_pow - 1
            f = (tri(xe, mod) - (p % mod) * tri(xnext, mod)) % mod
            total = (total + (A % mod) * f) % mod
            p_pow *= p
            pe *= p

    # Part 2: primes p > sqrt(N), only e=1. Group by q=floor(N/p) (< sqrtN).
    for q in range(1, sqrtN):
        hi = N // q
        if hi <= sqrtN:
            break

        lo = N // (q + 1) + 1
        if lo <= sqrtN:
            lo = sqrtN + 1
        if lo > hi:
            continue

        sum_p = (psum(hi) - psum(lo - 1)) % mod
        cnt_p = pi(hi) - pi(lo - 1)
        total = (total + tri(q, mod) * ((sum_p - (2 * (cnt_p % mod))) % mod)) % mod

    return total % mod


def main() -> None:
    # Asserts for the test values given in the problem statement.
    assert t_of_n(45, sieve_primes(100)) == 52
    assert T_naive(10) == 26
    assert T_naive(100) == 5282

    print(compute_T_mod(TARGET, MOD))


if __name__ == "__main__":
    main()
