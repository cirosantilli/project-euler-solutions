#!/usr/bin/env python3
"""
Project Euler 830 — Binomials and Powers

Let S(n) = sum_{k=0..n} C(n,k) * k^n.
Compute S(10^18) modulo 83^3 * 89^3 * 97^3.

No external libraries are used.
"""

from __future__ import annotations


PRIMES = (83, 89, 97)
POWER = 3


def egcd(a: int, b: int) -> tuple[int, int, int]:
    """Extended gcd: returns (g, x, y) with ax + by = g = gcd(a,b)."""
    x0, y0, x1, y1 = 1, 0, 0, 1
    while b:
        q = a // b
        a, b = b, a - q * b
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    return a, x0, y0


def modinv(a: int, m: int) -> int:
    """Modular inverse of a modulo m (m>1), assuming gcd(a,m)=1."""
    g, x, _ = egcd(a % m, m)
    if g != 1:
        raise ValueError("inverse does not exist")
    return x % m


def v_p(x: int, p: int) -> int:
    """p-adic valuation v_p(x) for x>=0. For x=0 return a large sentinel."""
    if x == 0:
        return 10**18
    c = 0
    while x % p == 0:
        x //= p
        c += 1
    return c


def max_j_before_power_divides_falling_factorial(n: int, p: int, a: int) -> int:
    """
    Returns the largest J such that v_p(n^(underline J)) < a,
    where n^(underline J) = n*(n-1)*...*(n-J+1).
    For J>n, the falling factorial is 0, hence divisible by p^a.
    """
    if n <= 0:
        return 0

    vp = 0
    j = 0
    while True:
        if vp >= a:
            return j - 1
        j += 1
        if j > n:
            return n
        vp += v_p(n - j + 1, p)


def forward_differences_of_powers(n: int, J: int, mod: int) -> list[int]:
    """
    Computes F[j] = Δ^j f(0) modulo mod for f(i) = i^n, 0<=j<=J,
    using an in-place forward-difference table.

    Identity used: Δ^j (i^n)|_{i=0} = sum_{t=0..j} (-1)^(j-t) * C(j,t) * t^n = j! * S(n,j),
    where S(n,j) is a Stirling number of the second kind.
    """
    arr = [0] * (J + 1)
    if n == 0:
        # 0^0 is taken as 1 in the original sum; here f(0)=0^0=1, f(i)=i^0=1 for i>0.
        for i in range(J + 1):
            arr[i] = 1 % mod
    else:
        arr[0] = 0
        for i in range(1, J + 1):
            arr[i] = pow(i, n, mod)

    F = [0] * (J + 1)
    for j in range(J + 1):
        F[j] = arr[0] % mod
        # next difference level
        for i in range(J - j):
            arr[i] = (arr[i + 1] - arr[i]) % mod
    return F


def binom_mod_prefix(n: int, J: int, mod: int) -> list[int]:
    """Returns [C(n,0), ..., C(n,J)] modulo mod. (Exact integer update; J is small.)"""
    out = [0] * (J + 1)
    c = 1
    out[0] = 1 % mod
    for j in range(1, J + 1):
        c = (c * (n - j + 1)) // j
        out[j] = c % mod
    return out


def solve_mod_prime_power(n: int, p: int, a: int) -> int:
    """
    Computes S(n) modulo p^a (here a=3), using:
      S(n) = sum_{j=0..n} C(n,j) * (j! * S(n,j)) * 2^(n-j)
    and truncation: if v_p(n^(underline j)) >= a then the term is 0 mod p^a.
    """
    mod = p**a
    if n == 0:
        return 1 % mod

    J = max_j_before_power_divides_falling_factorial(n, p, a)
    J = min(J, n)

    # j! * S(n,j) modulo p^a via forward differences
    fact_stirling = forward_differences_of_powers(n, J, mod)

    # C(n,j) modulo p^a
    choose = binom_mod_prefix(n, J, mod)

    pow2 = pow(2, n, mod)
    inv2 = modinv(2, mod)

    res = 0
    cur_pow2 = pow2
    for j in range(J + 1):
        res = (res + choose[j] * fact_stirling[j] % mod * cur_pow2) % mod
        cur_pow2 = (cur_pow2 * inv2) % mod  # 2^(n-(j+1))
    return res


def crt_pairwise(residues: list[int], moduli: list[int]) -> int:
    """Chinese Remainder Theorem for pairwise-coprime moduli."""
    M = 1
    for m in moduli:
        M *= m

    x = 0
    for r, m in zip(residues, moduli):
        Mi = M // m
        inv = modinv(Mi % m, m)
        x = (x + r * Mi * inv) % M
    return x


def solve(n: int) -> int:
    mods = [p**POWER for p in PRIMES]
    residues = [solve_mod_prime_power(n, p, POWER) for p in PRIMES]
    return crt_pairwise(residues, mods)


def main() -> None:
    # Test value given in the problem statement
    assert solve(10) == 142469423360

    n = 10**18
    print(solve(n))


if __name__ == "__main__":
    main()
