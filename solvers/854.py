#!/usr/bin/env python3
"""Project Euler 854: Pisano Periods 2

We need P(N) = \prod_{p=1..N} M(p) (mod MOD), where M(p) is the largest n with
Pisano period pi(n)=p, or 1 if no such n exists.

Key facts used (sketched in README):
- For n>2, pi(n) is even, so M(p)=1 for all odd p>3.
- The maximal modulus whose Pisano period divides p is
      N(p) = gcd(F_p, F_{p+1}-1),
  where F_k is the k-th Fibonacci number.
- For even p=2m (m>=3):
      N(2m) = F_m   if m is even,
      N(2m) = L_m   if m is odd,
  where L_m is the m-th Lucas number.
  Moreover these values actually have Pisano period 2m, so M(2m)=N(2m).

Therefore, for N=1_000_000:
  P(N) = M(3) * \prod_{m=3..N/2} (F_m if m even else L_m)  (mod MOD).

No external libraries are used.
"""

from __future__ import annotations

MOD = 1_234_567_891
TARGET = 1_000_000


def _fib_pair(n: int) -> tuple[int, int]:
    """Return (F_n, F_{n+1}) using fast doubling (exact integers)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return (0, 1)
    a, b = _fib_pair(n >> 1)
    c = a * ((b << 1) - a)
    d = a * a + b * b
    if n & 1:
        return (d, c + d)
    return (c, d)


def fib(n: int) -> int:
    """Exact Fibonacci number F_n."""
    return _fib_pair(n)[0]


def lucas(n: int) -> int:
    """Exact Lucas number L_n, computed from Fibonacci numbers."""
    fn, fn1 = _fib_pair(n)
    # L_n = 2*F_{n+1} - F_n
    return 2 * fn1 - fn


def M(p: int) -> int:
    """Largest modulus with Pisano period exactly p (exact integer for small p)."""
    if p == 3:
        return 2
    if p % 2 == 1:
        return 1
    m = p // 2
    if m < 3:
        return 1
    return fib(m) if (m % 2 == 0) else lucas(m)


def P_small(n: int) -> int:
    """Exact product P(n) for small n (used only for asserts)."""
    prod = 1
    for p in range(1, n + 1):
        prod *= M(p)
    return prod


def solve(limit_p: int = TARGET, mod: int = MOD) -> int:
    """Compute P(limit_p) mod mod."""
    if limit_p <= 0:
        return 1 % mod

    res = 1
    if limit_p >= 3:
        res = 2 % mod  # M(3)

    max_m = limit_p // 2

    # Iterate Fibonacci and Lucas sequences modulo mod up to index max_m.
    f_prev, f_cur = 0, 1  # F_0, F_1
    l_prev, l_cur = 2 % mod, 1 % mod  # L_0, L_1

    for m in range(2, max_m + 1):
        f_prev, f_cur = f_cur, (f_prev + f_cur) % mod
        l_prev, l_cur = l_cur, (l_prev + l_cur) % mod

        if m < 3:
            continue

        if m % 2 == 0:
            res = (res * f_cur) % mod
        else:
            res = (res * l_cur) % mod

    return res


def _self_test() -> None:
    # Test values explicitly given in the problem statement.
    assert M(18) == 76
    assert P_small(10) == 264


def main() -> None:
    _self_test()
    print(solve())


if __name__ == "__main__":
    main()
