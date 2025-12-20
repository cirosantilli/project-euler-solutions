#!/usr/bin/env python3
"""Project Euler 881: Divisor Graph Width

We model the divisor graph in terms of the exponent vector of n.
If n = \prod p_i^{e_i}, every divisor corresponds to a vector (f_i) with 0<=f_i<=e_i.
An edge changes exactly one coordinate by 1, so distance from n to a divisor is the
sum of exponent deficits. Therefore the number of vertices at level k is the
coefficient of x^k in \prod (1 + x + ... + x^{e_i}).

So g(n) is the maximum coefficient of that product.

We search over non-increasing exponent sequences (to minimize n by assigning larger
exponents to smaller primes) and use a branch-and-bound DFS to find the smallest n
with g(n) >= 10^4.

No external libraries are used.
"""

from __future__ import annotations


def sieve_primes(limit: int) -> list[int]:
    """Simple sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[0:2] = b"\x00\x00"

    # integer sqrt without importing math
    r = int(limit**0.5)
    while (r + 1) * (r + 1) <= limit:
        r += 1
    while r * r > limit:
        r -= 1

    for p in range(2, r + 1):
        if sieve[p]:
            start = p * p
            step = p
            sieve[start : limit + 1 : step] = b"\x00" * (((limit - start) // step) + 1)
    return [i for i in range(limit + 1) if sieve[i]]


PRIMES = sieve_primes(400)


def convolve_with_ones(poly: list[int], e: int) -> list[int]:
    """Return poly * (1 + x + ... + x^e).

    Since the second factor has all coefficients 1, this is a sliding-window sum.
    Runs in O(len(poly) + e) time.
    """
    if e < 0:
        raise ValueError("e must be non-negative")
    n = len(poly)
    pref = [0] * (n + 1)
    s = 0
    for i, v in enumerate(poly):
        s += v
        pref[i + 1] = s

    out_len = n + e
    out = [0] * out_len
    for j in range(out_len):
        lo = j - e
        if lo < 0:
            lo = 0
        hi = j
        if hi >= n:
            hi = n - 1
        if lo <= hi:
            out[j] = pref[hi + 1] - pref[lo]
    return out


def max_level_width_from_exponents(exps: list[int]) -> int:
    """Compute g(n) given the exponent list of n."""
    poly = [1]
    for e in exps:
        poly = convolve_with_ones(poly, e)
    return max(poly)


def factor_exponents(n: int) -> list[int]:
    """Return prime exponents of n (sorted descending)."""
    if n <= 0:
        raise ValueError("n must be positive")
    exps: list[int] = []
    x = n
    for p in PRIMES:
        if p * p > x:
            break
        if x % p == 0:
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            exps.append(e)
    if x > 1:
        exps.append(1)
    exps.sort(reverse=True)
    return exps


def g_of_n(n: int) -> int:
    """Compute g(n) by factoring n and using the polynomial model."""
    return max_level_width_from_exponents(factor_exponents(n))


def initial_upper_bound(target: int) -> int:
    """A quick upper bound using n as a product of distinct primes.

    With all exponents 1, the polynomial is (1+x)^m and the maximum coefficient is
    the central binomial coefficient. We increase m until we hit the target.
    """
    poly = [1]
    n = 1
    i = 0
    while True:
        poly = convolve_with_ones(poly, 1)
        n *= PRIMES[i]
        i += 1
        if max(poly) >= target:
            return n


def solve(target: int = 10_000) -> int:
    """Find the smallest n such that g(n) >= target."""

    # Given examples from the statement
    assert g_of_n(45) == 2
    assert g_of_n(5040) == 12

    best_n = initial_upper_bound(target)
    best_exps: tuple[int, ...] | None = None

    def dfs(
        idx: int, prev_e: int, poly: list[int], peak: int, n: int, exps: list[int]
    ) -> None:
        nonlocal best_n, best_exps

        if n >= best_n:
            return
        if peak >= target:
            best_n = n
            best_exps = tuple(exps)
            return
        if idx >= len(PRIMES):
            return

        p = PRIMES[idx]

        # Maximum exponent allowed both by monotonicity and by keeping n < best_n.
        # We avoid logs by multiplying until we would exceed the bound.
        max_e = 0
        t = n
        while max_e < prev_e:
            t *= p
            if t >= best_n:
                break
            max_e += 1

        if max_e == 0:
            return

        # Heuristic: try larger exponents first to reach the target sooner and tighten best_n.
        # Correctness does not depend on this ordering.
        ppow = 1
        powers = [1] * (max_e + 1)
        for e in range(1, max_e + 1):
            ppow *= p
            powers[e] = ppow

        for e in range(max_e, 0, -1):
            n2 = n * powers[e]
            if n2 >= best_n:
                continue
            poly2 = convolve_with_ones(poly, e)
            peak2 = max(poly2)
            dfs(idx + 1, e, poly2, peak2, n2, exps + [e])

    # Start with a generous maximum exponent for 2 (it will be bounded by best_n anyway).
    dfs(0, 60, [1], 1, 1, [])

    # best_exps is kept for debugging; not printed to avoid spoilers.
    return best_n


def main() -> None:
    print(solve(10_000))


if __name__ == "__main__":
    main()
