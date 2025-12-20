#!/usr/bin/env python3
"""Project Euler 809: Rational Recurrence Relation

We need f(22/7) (mod 10^15) for the function:

    f(x) = x                                  if x is integral
         = f(1/(1-x))                         if x < 1
         = f(1/(ceil(x)-x) - 1 + f(x-1))      otherwise

The solution below avoids evaluating f(x) directly (values explode).
It uses the fact that for inputs of the form t + 1/n (t integer, n>=2),
this recurrence matches the classical Ackermann–Péter function.

No external libraries are used.
"""

from __future__ import annotations

from math import gcd


# ---------- basic number theory helpers ----------


def egcd(a: int, b: int) -> tuple[int, int, int]:
    """Extended GCD: returns (g, x, y) such that a*x + b*y = g = gcd(a,b)."""
    if b == 0:
        return (a, 1, 0)
    g, x1, y1 = egcd(b, a % b)
    return (g, y1, x1 - (a // b) * y1)


def inv_mod(a: int, m: int) -> int:
    """Modular inverse of a modulo m (requires gcd(a,m)=1)."""
    g, x, _ = egcd(a, m)
    if g != 1:
        raise ValueError("inverse does not exist")
    return x % m


def crt(r1: int, m1: int, r2: int, m2: int) -> int:
    """Chinese Remainder Theorem for coprime moduli.

    Returns x modulo m1*m2 such that:
        x ≡ r1 (mod m1)
        x ≡ r2 (mod m2)
    """
    # x = r1 + m1 * k
    # r1 + m1*k ≡ r2 (mod m2)  =>  k ≡ (r2-r1) * inv(m1) (mod m2)
    k = ((r2 - r1) % m2) * inv_mod(m1 % m2, m2) % m2
    return r1 + m1 * k


# ---------- fast totient for numbers of the form 2^a * 5^b ----------


def factor_2_5(n: int) -> tuple[int, int, int]:
    """Return (a, b, rem) where n = 2^a * 5^b * rem and rem is not divisible by 2 or 5."""
    a = 0
    while n % 2 == 0:
        a += 1
        n //= 2
    b = 0
    while n % 5 == 0:
        b += 1
        n //= 5
    return a, b, n


def phi_2_5(n: int) -> int:
    """Euler's totient for n containing only primes 2 and 5."""
    a, b, rem = factor_2_5(n)
    if rem != 1:
        raise ValueError("phi_2_5 expects n to factor only into 2s and 5s")
    res = n
    if a:
        res //= 2
    if b:
        res = (res // 5) * 4
    return res


def totient_chain_len(n: int) -> int:
    """How many times do we need to apply phi until we reach 1?"""
    steps = 0
    while n != 1:
        n = phi_2_5(n)
        steps += 1
    return steps


# ---------- 2-tetration modulo 2^a and 2^a*5^b ----------


def tetration_cap(height: int, cap: int) -> int:
    """Return min(2↑↑height, cap) for base=2.

    Only used with very small caps (<= 2^16 in this problem), so a simple loop is fine.
    """
    if cap <= 0:
        return 0
    v = 2
    if height <= 1:
        return v if v < cap else cap

    for _ in range(2, height + 1):
        # Next term is 2^v. If v is already large enough, we know we hit the cap.
        if v >= 60:  # 2^60 is enormous compared to all caps we need
            return cap
        v = 1 << v
        if v >= cap:
            return cap
    return v


def tetration_mod_pow2(height: int, a: int) -> int:
    """Compute 2↑↑height (mod 2^a)."""
    if a <= 0:
        return 0
    mod = 1 << a
    if height == 1:
        return 2 % mod

    # 2↑↑height = 2^(2↑↑(height-1)) is always a power of two.
    exp = tetration_cap(height - 1, a)
    if exp >= a:
        return 0
    return (1 << exp) % mod


def tetration_mod(height: int, mod: int) -> int:
    """Compute 2↑↑height (mod mod) for mod of the form 2^a*5^b."""
    if mod == 1:
        return 0

    a, b, rem = factor_2_5(mod)
    if rem != 1:
        raise ValueError("This solver only needs moduli with prime factors 2 and 5")

    if b == 0:
        return tetration_mod_pow2(height, a)

    if a == 0:
        # Odd modulus (5^b): gcd(2,mod)=1, so reducing the exponent mod phi(mod) is valid.
        if height == 1:
            return 2 % mod
        exp = tetration_mod(height - 1, phi_2_5(mod))
        return pow(2, exp, mod)

    # Mixed modulus: solve mod 2^a and mod 5^b separately and combine by CRT.
    m2 = 1 << a
    m5 = 5**b
    r2 = tetration_mod_pow2(height, a)
    r5 = tetration_mod(height, m5)
    return crt(r2, m2, r5, m5) % (m2 * m5)


def stable_tetration_mod(mod: int) -> int:
    """Compute 2↑↑H (mod mod) for a height H that is 'tall enough'.

    If we choose H = totient_chain_len(mod) + 1, the recursion reaches mod==1
    before it reaches the height base case, so any larger H gives the same residue.
    """
    height = totient_chain_len(mod) + 1
    return tetration_mod(height, mod)


# ---------- small helpers for the statement's sample values ----------


def ackermann_base2_small(m: int, n: int) -> int:
    """Ackermann–Péter A(m,n) for base-case A(0,n)=n+1, specialized for small m,n.

    Only used for asserts from the statement.
    """
    if m == 0:
        return n + 1
    if m == 1:
        return n + 2
    if m == 2:
        return 2 * n + 3
    if m == 3:
        return (1 << (n + 3)) - 3
    if m == 4:
        # A(4,0)=A(3,1)=13
        v = ackermann_base2_small(3, 1)
        for _ in range(n):
            v = (1 << (v + 3)) - 3
        return v
    if m == 5 and n == 0:
        # A(5,0)=A(4,1)
        return ackermann_base2_small(4, 1)

    raise ValueError("This helper is intentionally limited")


def f_shallow(p: int, q: int) -> int:
    """Direct evaluation of f(p/q) for very small inputs.

    Uses an explicit stack to avoid recursion depth issues.
    This is *not* intended for the main solve; only for the statement's examples.
    """
    memo: dict[tuple[int, int], int] = {}

    def norm(a: int, b: int) -> tuple[int, int]:
        g = gcd(a, b)
        return a // g, b // g

    p, q = norm(p, q)
    stack: list[tuple[int, int, int, int, int]] = [(p, q, 0, 0, 0)]
    # frame: (p, q, stage, aux1, aux2)
    # stage 0: expand
    # stage 1: after single-child transform
    # stage 2: after computing f(p-q,q)
    # stage 3: after computing f(num,den)

    while stack:
        p, q, stage, aux1, aux2 = stack.pop()
        p, q = norm(p, q)
        key = (p, q)

        if key in memo:
            continue

        if q == 1:
            memo[key] = p
            continue

        if p < q:
            child = norm(q, q - p)
            if child in memo:
                memo[key] = memo[child]
            else:
                stack.append((p, q, 1, child[0], child[1]))
                stack.append((child[0], child[1], 0, 0, 0))
            continue

        if p % q == 0:
            memo[key] = p // q
            continue

        if stage == 0:
            # need m = f(p-q, q)
            child = norm(p - q, q)
            if child in memo:
                m = memo[child]
                k = p // q
                rem = p - k * q
                num = m * (q - rem) + rem
                den = q - rem
                child2 = norm(num, den)
                if child2 in memo:
                    memo[key] = memo[child2]
                else:
                    stack.append((p, q, 3, child2[0], child2[1]))
                    stack.append((child2[0], child2[1], 0, 0, 0))
            else:
                stack.append((p, q, 2, child[0], child[1]))
                stack.append((child[0], child[1], 0, 0, 0))
            continue

        if stage == 1:
            child = (aux1, aux2)
            memo[key] = memo[child]
            continue

        if stage == 2:
            # child = f(p-q, q) is ready
            child = (aux1, aux2)
            m = memo[child]
            k = p // q
            rem = p - k * q
            num = m * (q - rem) + rem
            den = q - rem
            child2 = norm(num, den)
            if child2 in memo:
                memo[key] = memo[child2]
            else:
                stack.append((p, q, 3, child2[0], child2[1]))
                stack.append((child2[0], child2[1], 0, 0, 0))
            continue

        if stage == 3:
            child2 = (aux1, aux2)
            memo[key] = memo[child2]
            continue

        raise RuntimeError("unreachable")

    return memo[(p, q)]


def example_f_13_10() -> int:
    """Compute f(13/10) using only small f evaluations and a short linear recurrence."""
    # f(13/10) = f(1 + 3/10)
    # Let g(m) = f(m + 3/7). Then g(0)=f(3/7) and g(m+1)=3*g(m)+4.
    # Also f(3/10)=25, so f(13/10)=f(25 + 3/7)=g(25).
    m = f_shallow(3, 10)  # 25
    g0 = f_shallow(3, 7)  # 7
    v = g0
    for _ in range(m):
        v = 3 * v + 4
    return v


# ---------- solve ----------


def solve() -> int:
    # f(22/7) = f(3 + 1/7) = A(6,3).
    # A(6,3) + 3 = 2 ↑↑↑↑ 6. Mod 5^15 this stabilizes to a tall tetration residue.
    mod2 = 1 << 15
    mod5 = 5**15

    # 2 ↑↑↑↑ 6 is a power of two, so mod 2^15 it is 0, hence A(6,3) ≡ -3.
    r2 = (-3) % mod2

    # For the odd part we compute a sufficiently tall power tower of 2's modulo 5^15.
    tower_mod5 = stable_tetration_mod(mod5)
    r5 = (tower_mod5 - 3) % mod5

    return crt(r2, mod2, r5, mod5) % (10**15)


def _self_test() -> None:
    # Values given in the problem statement
    assert ackermann_base2_small(1, 1) == 3  # f(3/2) = f(1+1/2)
    assert ackermann_base2_small(5, 0) == 65533  # f(1/6)
    assert example_f_13_10() == 7625597484985  # f(13/10)


if __name__ == "__main__":
    _self_test()
    print(solve())
