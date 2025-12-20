#!/usr/bin/env python3
"""
Project Euler 513 — Integral median

Count integer-sided triangles a <= b <= c (with c <= n) whose median to side c
is also an integer.

Key facts used:
- Median formula: m^2 = (2a^2 + 2b^2 - c^2) / 4
- Necessarily c is even (otherwise numerator can't be divisible by 4).
- A change of variables turns the condition into a difference-of-squares identity,
  and then into a factor/gcd structure that can be counted efficiently.

No external libraries are used; single core; no multithreading.
"""

import sys


def mobius_sieve(n: int):
    """Compute Möbius mu[1..n] in O(n) using a linear sieve."""
    mu = [0] * (n + 1)
    mu[1] = 1
    primes = []
    is_comp = [False] * (n + 1)
    for i in range(2, n + 1):
        if not is_comp[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            ip = i * p
            if ip > n:
                break
            is_comp[ip] = True
            if i % p == 0:
                mu[ip] = 0
                break
            mu[ip] = -mu[i]
    return mu


def build_squarefree_divisor_lists(n: int, mu):
    """
    For every A <= n, build:
      - all squarefree divisors d of A (i.e. mu[d] != 0) with their mu[d]
      - odd squarefree divisors similarly
    using a divisor-sieve style construction.
    """
    all_divs = [[] for _ in range(n + 1)]
    odd_divs = [[] for _ in range(n + 1)]

    for d in range(1, n + 1):
        md = mu[d]
        if md == 0:
            continue
        if d & 1:
            for m in range(d, n + 1, d):
                all_divs[m].append((d, md))
                odd_divs[m].append((d, md))
        else:
            for m in range(d, n + 1, d):
                all_divs[m].append((d, md))

    return all_divs, odd_divs


def count_k_with_parity(lo: int, hi: int, parity: int) -> int:
    """Count integers k in [lo, hi] such that k % 2 == parity."""
    if hi < lo:
        return 0
    if (lo & 1) != parity:
        lo += 1
    if lo > hi:
        return 0
    return 1 + ((hi - lo) >> 1)


def coprime_count_interval(divs, L: int, U: int) -> int:
    """
    Count B in [L, U] with gcd(A, B) = 1 using inclusion-exclusion:
        count = sum_{d|A} mu[d] * (floor(U/d) - floor((L-1)/d))
    where divs is list of (d, mu[d]) for squarefree d|A.
    """
    if U < L:
        return 0
    Lm1 = L - 1
    s = 0
    for d, md in divs:
        s += md * (U // d - Lm1 // d)
    return s


def coprime_count_interval_odd(odd_divs, L: int, U: int) -> int:
    """
    Count odd B in [L, U] with gcd(A, B) = 1.
    For odd d, the odd multiples of d up to X are:
        count_odd_multiples(X, d) = (floor(X/d) + 1) // 2
    Thus:
        f(X) = sum_{odd d|A} mu[d] * ((X//d + 1)//2)
        answer = f(U) - f(L-1)
    odd_divs is list of (d, mu[d]) for odd squarefree d|A.
    """
    if U < L:
        return 0

    def f(X: int) -> int:
        if X <= 0:
            return 0
        s = 0
        for d, md in odd_divs:
            s += md * ((X // d + 1) >> 1)
        return s

    return f(U) - f(L - 1)


def F(n: int) -> int:
    """
    Compute F(n) for the Euler 513 definition.

    Sketch of the counting approach:
    - Let c = 2C (so C is integer).
    - Use x=(b-a)/2, y=(a+b)/2 (integers because a,b same parity under solutions).
    - Condition becomes: C^2 + m^2 = x^2 + y^2, equivalent to:
          (C-x)(C+x) = (y-m)(y+m)
    - Let p = C-x and q = y-m. From gcd arguments:
          C+x = k*(q/g),  y+m = k*(p/g), where g=gcd(p,q).
    - After enforcing ordering and positivity for the median, it turns out we only
      need the regime k > g and (q/g) < (p/g).
    - Rename A = p/g and B = q/g with gcd(A,B)=1 and B < A.
      Then c = g*A + k*B and constraints yield a k-interval for each (A,g).
    - For a fixed (A,g), k is counted in blocks where floor divisions are constant.
      For each k-block we count admissible B via Möbius inclusion-exclusion
      over an interval [Bmin, Bmax].

    This runs comfortably for n=100000 in optimized Python.
    """
    maxA = n // 2
    mu = mobius_sieve(maxA)
    all_divs, odd_divs = build_squarefree_divisor_lists(maxA, mu)

    total = 0

    for A in range(2, maxA + 1):
        A_minus_1 = A - 1
        A_is_odd = A & 1

        divsA = all_divs[A]
        oddDivsA = odd_divs[A]

        # Since k must be > d and k ≡ d (mod 2), the smallest possible k is d+2.
        # With B >= 1, c = d*A + k*B >= d*A + (d+2)*1 = d*(A+1) + 2 <= n
        # => d <= (n-2)//(A+1)
        maxd = (n - 2) // (A + 1)
        if maxd <= 0:
            break

        for d in range(1, maxd + 1):
            parity = d & 1
            if parity and not A_is_odd:
                # If d is odd, parity constraints force A and B to be odd too.
                continue

            M1 = d * A
            M2 = n - M1
            # M2 >= d+2 by construction

            if parity:
                divlist = oddDivsA
                countB_interval = coprime_count_interval_odd
            else:
                divlist = divsA
                countB_interval = coprime_count_interval

            # k ranges:
            # 1) d+2 <= k <= min(3d, M2)   (no extra b<=c restriction)
            # 2) k >= 3d+2                (extra lower bound near A)
            #
            # Additionally we always require B < A and kB >= dA  => B >= ceil((dA)/k)

            k_start = d + 2

            # ---- Region 1: k <= 3d ----
            K2 = 3 * d
            if K2 > M2:
                K2 = M2
            if K2 >= k_start:
                k = k_start
                M1m1 = M1 - 1
                while k <= K2:
                    qmin = M1m1 // k
                    Bmin = qmin + 1
                    qmax = M2 // k
                    Bmax = qmax
                    if Bmax > A_minus_1:
                        Bmax = A_minus_1
                    if Bmin < 1:
                        Bmin = 1

                    if Bmax >= Bmin:
                        cntB = countB_interval(divlist, Bmin, Bmax)
                    else:
                        cntB = 0

                    # end of block where qmin or qmax changes
                    if qmin:
                        kend1 = M1m1 // qmin
                    else:
                        kend1 = K2
                    kend2 = M2 // qmax  # qmax >= 1
                    kend = kend1 if kend1 < kend2 else kend2
                    if kend > K2:
                        kend = K2

                    cntK = count_k_with_parity(k, kend, parity)
                    if cntB and cntK:
                        total += cntB * cntK

                    k = kend + 1

            # ---- Region 2: k > 3d ----
            # Note: 3d has same parity as d, so the first k with required parity is 3d+2.
            k = 3 * d + 2
            if k <= M2:
                M1m1 = M1 - 1
                twoAd = 2 * A * d
                while k <= M2:
                    t = k - d
                    qextra = twoAd // t
                    if qextra == 0:
                        break
                    # b<=c constraint transforms to:
                    #   B >= A - floor(2Ad / (k-d)) = A - qextra
                    Bmin2 = A - qextra

                    qmin = M1m1 // k
                    Bmin = qmin + 1
                    if Bmin < 1:
                        Bmin = 1
                    if Bmin < Bmin2:
                        Bmin = Bmin2

                    qmax = M2 // k
                    Bmax = qmax
                    if Bmax > A_minus_1:
                        Bmax = A_minus_1

                    if Bmax >= Bmin:
                        cntB = countB_interval(divlist, Bmin, Bmax)
                    else:
                        cntB = 0

                    # end of block where qmin/qmax/qextra changes
                    kend1 = (M1m1 // qmin) if qmin else M2
                    kend2 = M2 // qmax
                    tmax = twoAd // qextra
                    kend3 = d + tmax

                    kend = kend1
                    if kend2 < kend:
                        kend = kend2
                    if kend3 < kend:
                        kend = kend3
                    if kend > M2:
                        kend = M2

                    cntK = count_k_with_parity(k, kend, parity)
                    if cntB and cntK:
                        total += cntB * cntK

                    k = kend + 1

    return total


def main():
    # Problem statement test values
    assert F(10) == 3
    assert F(50) == 165

    n = 100000
    print(F(n))


if __name__ == "__main__":
    main()
