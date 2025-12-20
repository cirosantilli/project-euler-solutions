#!/usr/bin/env python3
"""Project Euler 834: Add and Divide

We define a sequence starting from n, and at the m-th step we add (n+m).
The m-th term is divisible by (n+m) for exactly those m in S(n).

This script computes U(1234567) = sum_{n=3..1234567} T(n),
where T(n) is the sum of indices in S(n).

It also self-checks the example values from the problem statement.

No external (third-party) libraries are used.
"""

from bisect import bisect_right


def build_spf(limit: int) -> list[int]:
    """Return a smallest-prime-factor table for 0..limit (linear sieve)."""
    spf = [0] * (limit + 1)
    primes: list[int] = []
    for i in range(2, limit + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
        for p in primes:
            ip = i * p
            if ip > limit:
                break
            spf[ip] = p
            if p == spf[i]:
                break
    spf[0] = 0
    if limit >= 1:
        spf[1] = 1
    return spf


def odd_part_and_twopow(x: int) -> tuple[int, int]:
    """Return (odd_part(x), 2^v2(x)) for x>0."""
    two = x & -x
    return x // two, two


def divisors_with_prefix(
    odd_x: int, spf: list[int]
) -> tuple[list[int], list[int], int]:
    """Return (sorted_divisors, prefix_sums, total_sum) for an odd positive integer."""
    if odd_x == 1:
        divs = [1]
        pre = [0, 1]
        return divs, pre, 1

    # Factorization via SPF
    factors: list[tuple[int, int]] = []
    x = odd_x
    while x > 1:
        p = spf[x]
        e = 0
        while x % p == 0:
            x //= p
            e += 1
        factors.append((p, e))

    # Generate divisors
    divs = [1]
    for p, e in factors:
        base = divs[:]  # current divisors
        mult = 1
        for _ in range(e):
            mult *= p
            for d in base:
                divs.append(d * mult)

    divs.sort()

    pre = [0] * (len(divs) + 1)
    s = 0
    for i, v in enumerate(divs, 1):
        s += v
        pre[i] = s
    return divs, pre, s


def count_sum_products_gt(
    small_divs: list[int],
    large_divs: list[int],
    large_prefix: list[int],
    large_total: int,
    bound: int,
) -> tuple[int, int]:
    """Over pairs (a in small_divs, b in large_divs), compute:

    count = #{ a*b > bound }
    s = sum_{a*b > bound} a*b

    large_divs must be sorted ascending and large_prefix its prefix sums.
    """
    n_large = len(large_divs)
    cnt = 0
    s = 0

    # Local bindings for speed
    bd = bound
    ldiv = large_divs
    lpre = large_prefix
    ltot = large_total
    br = bisect_right

    for a in small_divs:
        idx = br(ldiv, bd // a)  # want b > bound//a
        if idx != n_large:
            cnt += n_large - idx
            s += a * (ltot - lpre[idx])
    return cnt, s


def compute_T_from_divdata(n: int, two: int, dataA, dataB) -> int:
    """Compute T(n) given divisor data for the odd factors A,B and twopow(two)."""
    divA, preA, sumA = dataA
    divB, preB, sumB = dataB

    # Products p = a*b where a|A, b|B represent all odd divisors of A*B.
    # Valid (n+m)=d are:
    #   d = p (odd) when p > n
    #   d = two*p (even) when two*p > n  <=>  p > n//two
    # Then T(n) = sum_{valid d} (d - n).

    if len(divA) <= len(divB):
        small, large, preL, sumL = divA, divB, preB, sumB
    else:
        small, large, preL, sumL = divB, divA, preA, sumA

    cnt1, sum1 = count_sum_products_gt(small, large, preL, sumL, n)
    cnt2, sum2 = count_sum_products_gt(small, large, preL, sumL, n // two)

    count_total = cnt1 + cnt2
    sum_d = sum1 + two * sum2
    return sum_d - n * count_total


def compute_T_single(n: int, spf: list[int]) -> int:
    """Compute T(n) without relying on rolling caching (handy for tests)."""
    if n & 1:
        even = n - 1
        B = n
    else:
        even = n
        B = n - 1
    A, two = odd_part_and_twopow(even)
    dataA = divisors_with_prefix(A, spf)
    dataB = divisors_with_prefix(B, spf)
    return compute_T_from_divdata(n, two, dataA, dataB)


def compute_U(N: int) -> int:
    """Compute U(N) = sum_{n=3..N} T(n)."""
    spf = build_spf(N)

    # Rolling caches: for consecutive n, exactly one of A or B repeats.
    lastA = -1
    lastB = -1
    dataA = None
    dataB = None

    total = 0
    for n in range(3, N + 1):
        if n & 1:
            even = n - 1
            B = n
        else:
            even = n
            B = n - 1

        A, two = odd_part_and_twopow(even)

        if A != lastA:
            dataA = divisors_with_prefix(A, spf)
            lastA = A
        if B != lastB:
            dataB = divisors_with_prefix(B, spf)
            lastB = B

        # mypy: dataA/dataB are set before use
        total += compute_T_from_divdata(n, two, dataA, dataB)

    return total


def compute_S_bruteforce(n: int) -> list[int]:
    """Bruteforce S(n) by enumerating all relevant divisors (only for tiny n)."""
    spf = build_spf(max(3, n))

    if n & 1:
        even = n - 1
        B = n
    else:
        even = n
        B = n - 1
    A, two = odd_part_and_twopow(even)

    divA, _, _ = divisors_with_prefix(A, spf)
    divB, _, _ = divisors_with_prefix(B, spf)

    ms = []
    for a in divA:
        for b in divB:
            p = a * b
            if p > n:
                ms.append(p - n)
            if two * p > n:
                ms.append(two * p - n)

    ms.sort()
    out = []
    last = None
    for m in ms:
        if m != last:
            out.append(m)
            last = m
    return out


def _self_test() -> None:
    # Example set S(10)
    assert compute_S_bruteforce(10) == [5, 8, 20, 35, 80]

    # Example T values
    spf = build_spf(100)
    assert compute_T_single(10, spf) == 148
    assert compute_T_single(100, spf) == 21828

    # Example U(100)
    assert compute_U(100) == 612572


def main() -> None:
    _self_test()
    print(compute_U(1234567))


if __name__ == "__main__":
    main()
