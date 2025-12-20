#!/usr/bin/env python3
"""
Project Euler 786 - Billiard

Pure Python, single-threaded, no external dependencies.

The program:
- Asserts the sample values from the statement.
- Prints B(10^9).
"""

from __future__ import annotations


def floor_sum(n: int, m: int, a: int, b: int) -> int:
    """
    Returns sum_{i=0..n-1} floor((a*i + b) / m) for n >= 0, m > 0.
    Based on a standard iterative reduction (often used in competitive programming).
    """
    assert n >= 0 and m > 0
    res = 0
    while True:
        if a >= m:
            res += (n - 1) * n * (a // m) // 2
            a %= m
        if b >= m:
            res += n * (b // m)
            b %= m
        y_max = a * n + b
        if y_max < m:
            break
        n, b = divmod(y_max, m)
        m, a = a, m
    return res


def icbrt(n: int) -> int:
    """Integer cube root: largest x with x^3 <= n."""
    if n <= 0:
        return 0
    # Newton iteration, all-integer
    x = 1 << ((n.bit_length() + 2) // 3)  # rough power-of-two start
    while True:
        y = (2 * x + n // (x * x)) // 3
        if y >= x:
            break
        x = y
    while (x + 1) * (x + 1) * (x + 1) <= n:
        x += 1
    while x * x * x > n:
        x -= 1
    return x


def mobius_prefix(limit: int) -> list[int]:
    """
    Linear sieve for mu(k) prefix sums up to 'limit'.
    Returns pref where pref[n] = sum_{k<=n} mu(k).
    """
    mu = [0] * (limit + 1)
    is_comp = [False] * (limit + 1)
    primes: list[int] = []
    mu[1] = 1

    for i in range(2, limit + 1):
        if not is_comp[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            v = i * p
            if v > limit:
                break
            is_comp[v] = True
            if i % p == 0:
                mu[v] = 0
                break
            mu[v] = -mu[i]

    pref = [0] * (limit + 1)
    s = 0
    for i in range(1, limit + 1):
        s += mu[i]
        pref[i] = s
    return pref


def make_mertens(limit: int):
    """
    Builds M(n) = sum_{k<=n} mu(k) with a small sieve base up to 'limit'
    and a memoized recursion for larger n.
    Also provides F(n) = sum_{k<=n, 3 ∤ k} mu(k) via:
        M(n) = F(n) - F(floor(n/3))  =>  F(n) = M(n) + F(floor(n/3)).
    """
    pref = mobius_prefix(limit)
    cache_m: dict[int, int] = {}
    cache_f: dict[int, int] = {0: 0}

    def M(n: int) -> int:
        if n <= limit:
            return pref[n]
        v = cache_m.get(n)
        if v is not None:
            return v
        res = 1
        l = 2
        while l <= n:
            q = n // l
            r = n // q
            res -= (r - l + 1) * M(q)
            l = r + 1
        cache_m[n] = res
        return res

    def F(n: int) -> int:
        if n <= 0:
            return 0
        v = cache_f.get(n)
        if v is not None:
            return v
        res = M(n) + F(n // 3)
        cache_f[n] = res
        return res

    return M, F


def count_points_nonprimitive(M: int) -> int:
    """
    Count integer pairs (x,y) with x>=1, y>=1, 18x+10y <= M, and 3 ∤ y.
    This is the raw lattice-point count without the gcd(x,y)=1 constraint.
    """
    if M < 28:
        return 0

    # y max from 18*1 + 10*y <= M
    n = (M - 18) // 10

    # Sum_{y=1..n} floor((M - 10y)/18)  (x is at least 1 by construction)
    # Reverse-index trick to fit floor_sum form:
    # sum_{j=0..n-1} floor((10*j + (M - 10n))/18)
    b = M - 10 * n
    total = floor_sum(n, 18, 10, b)

    # Subtract y multiple of 3: y=3k
    n3 = n // 3  # equivalent to floor((M-18)/30)
    b3 = M - 30 * n3
    total3 = floor_sum(n3, 18, 30, b3)

    return total - total3


def count_points_primitive(M: int) -> int:
    """
    Count pairs (x,y) with x>=1, y>=1, 18x+10y <= M, gcd(x,y)=1, and 3 ∤ y.
    Uses Möbius inversion:
        sum_{d>=1, 3∤d} mu(d) * count_points_nonprimitive(floor(M/d))
    Grouping by constant floor(M/d) makes this ~O(sqrt(M)).
    """
    max_d = M // 28
    if max_d <= 0:
        return 0

    # A safe "small sieve" cut-off: ~ (max_d)^(2/3) = cbrt(max_d^2)
    limit = icbrt(max_d * max_d) + 64
    _, F = make_mertens(limit)

    ans = 0
    l = 1
    while l <= max_d:
        q = M // l
        r = M // q
        if r > max_d:
            r = max_d

        coef = F(r) - F(l - 1)  # sum_{d in [l..r], 3∤d} mu(d)
        if coef:
            ans += coef * count_points_nonprimitive(q)

        l = r + 1
    return ans


def B(N: int) -> int:
    """
    Number of valid billiard traces that start at A, bounce at most N times,
    and return to A (as defined in the problem).
    """
    M = 3 * N + 6
    return 2 + 4 * count_points_primitive(M)


def main() -> None:
    # Tests from the statement
    assert B(10) == 6
    assert B(100) == 478
    assert B(1000) == 45790

    print(B(10**9))


if __name__ == "__main__":
    main()
