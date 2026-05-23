#!/usr/bin/env python
"""
Project Euler 415: Titanic Sets

Count titanic subsets of the (N+1) by (N+1) lattice grid modulo 10^8.
The complement consists of the empty set, singletons, and collinear subsets.
"""


MOD = 100_000_000
DEFAULT_N = 100_000_000_000
PRECOMPUTE = 25_000_000


def norm(x: int) -> int:
    return x % MOD


def mul(a: int, b: int) -> int:
    return (a % MOD) * (b % MOD) % MOD


def s1(n: int) -> int:
    if n <= 0:
        return 0
    return n * (n + 1) // 2 % MOD


def s2(n: int) -> int:
    if n <= 0:
        return 0
    return n * (n + 1) * (2 * n + 1) // 6 % MOD


def s3(n: int) -> int:
    if n <= 0:
        return 0
    t = n * (n + 1) // 2
    return t * t % MOD


def range_s1(lo: int, hi: int) -> int:
    return (s1(hi) - s1(lo - 1)) % MOD


def range_s2(lo: int, hi: int) -> int:
    return (s2(hi) - s2(lo - 1)) % MOD


def range_s3(lo: int, hi: int) -> int:
    return (s3(hi) - s3(lo - 1)) % MOD


def pref_k_pow2(n: int, pow2_next: int) -> int:
    if n < 0:
        return 0
    return (((n - 1) % MOD) * pow2_next + 2) % MOD


def pref_k2_pow2(n: int, pow2_next: int) -> int:
    if n < 0:
        return 0
    k = n % MOD
    return ((k * k - 2 * k + 3) % MOD * pow2_next - 6) % MOD


class TotientSums:
    def __init__(self, max_n: int) -> None:
        limit = min(max_n, PRECOMPUTE)
        if limit < 1:
            limit = 1
        self.limit = limit

        phi = [0] * (limit + 1)
        phi[1] = 1
        primes: list[int] = []
        composite = bytearray(limit + 1)

        for x in range(2, limit + 1):
            if not composite[x]:
                primes.append(x)
                phi[x] = x - 1
            phix = phi[x]
            for p in primes:
                y = x * p
                if y > limit:
                    break
                composite[y] = 1
                if x % p == 0:
                    phi[y] = phix * p
                    break
                phi[y] = phix * (p - 1)

        pref0 = [0] * (limit + 1)
        pref1 = [0] * (limit + 1)
        pref2 = [0] * (limit + 1)
        for x in range(1, limit + 1):
            ph = phi[x] % MOD
            xm = x % MOD
            pref0[x] = (pref0[x - 1] + ph) % MOD
            pref1[x] = (pref1[x - 1] + xm * ph) % MOD
            pref2[x] = (pref2[x - 1] + xm * xm % MOD * ph) % MOD

        self.pref0 = pref0
        self.pref1 = pref1
        self.pref2 = pref2
        self.cache0: dict[int, int] = {}
        self.cache1: dict[int, int] = {}
        self.cache2: dict[int, int] = {}

    def phi(self, n: int) -> int:
        if n <= self.limit:
            return self.pref0[n]
        cached = self.cache0.get(n)
        if cached is not None:
            return cached

        total = s1(n)
        lo = 2
        while lo <= n:
            q = n // lo
            hi = n // q
            total = (total - ((hi - lo + 1) % MOD) * self.phi(q)) % MOD
            lo = hi + 1

        self.cache0[n] = total
        return total

    def i_phi(self, n: int) -> int:
        if n <= self.limit:
            return self.pref1[n]
        cached = self.cache1.get(n)
        if cached is not None:
            return cached

        total = s2(n)
        lo = 2
        while lo <= n:
            q = n // lo
            hi = n // q
            total = (total - range_s1(lo, hi) * self.i_phi(q)) % MOD
            lo = hi + 1

        self.cache1[n] = total
        return total

    def i2_phi(self, n: int) -> int:
        if n <= self.limit:
            return self.pref2[n]
        cached = self.cache2.get(n)
        if cached is not None:
            return cached

        total = s3(n)
        lo = 2
        while lo <= n:
            q = n // lo
            hi = n // q
            total = (total - range_s2(lo, hi) * self.i2_phi(q)) % MOD
            lo = hi + 1

        self.cache2[n] = total
        return total


def direction_stats(sums: TotientSums, m: int) -> tuple[int, int, int]:
    if m <= 0:
        return 0, 0, 0
    count = (2 * sums.phi(m) - 1) % MOD
    coord_sum = (3 * sums.i_phi(m) - 1) % MOD
    product_sum = sums.i2_phi(m)
    return count, coord_sum, product_sum


def titanic_sets(n: int, sums: TotientSums) -> int:
    side = n + 1
    point_count = side * side
    all_subsets = pow(2, point_count, MOD)
    singleton_part = (1 + (point_count % MOD)) % MOD

    if n < 2:
        return (all_subsets - singleton_part) % MOD

    blocks = []
    needed: set[int] = set()
    lo = 2
    while lo <= n:
        a = n // lo
        b = n // (lo + 1) if lo < n else 0
        hi_a = n // a
        hi_b = n // b - 1 if b else n
        hi = min(hi_a, hi_b)
        blocks.append((lo, hi, a, b))
        needed.add(a)
        if b:
            needed.add(b)
        lo = hi + 1

    stats = {m: direction_stats(sums, m) for m in needed}
    side_mod = side % MOD
    side2_mod = side_mod * side_mod % MOD

    collinear = 0
    for lo, hi, m1, m2 in blocks:
        c1, xy1, pr1 = stats[m1]
        c2, xy2, pr2 = stats[m2] if m2 else (0, 0, 0)

        q2 = (pr1 - pr2) % MOD
        q1 = (side_mod * (xy2 - xy1) - 2 * pr2) % MOD
        q0 = (side2_mod * (c1 - c2) + side_mod * xy2 - pr2) % MOD

        p2 = 2 * q2 % MOD
        p1 = 2 * q1 % MOD
        p0 = (2 * q0 + 2 * side_mod) % MOD

        pow_lo = pow(2, lo, MOD)
        pow_after_hi = pow(2, hi + 1, MOD)
        e0 = (pow_after_hi - pow_lo) % MOD
        e1 = (pref_k_pow2(hi, pow_after_hi) - pref_k_pow2(lo - 1, pow_lo)) % MOD
        e2 = (pref_k2_pow2(hi, pow_after_hi) - pref_k2_pow2(lo - 1, pow_lo)) % MOD

        poly_exp = (p2 * e2 + p1 * e1 + p0 * e0) % MOD

        r1 = range_s1(lo, hi)
        r2 = range_s2(lo, hi)
        r3 = range_s3(lo, hi)
        length = (hi - lo + 1) % MOD
        poly_plain = (
            p2 * ((r3 + r2) % MOD) + p1 * ((r2 + r1) % MOD) + p0 * ((r1 + length) % MOD)
        ) % MOD

        collinear = (collinear + poly_exp - poly_plain) % MOD

    return (all_subsets - singleton_part - collinear) % MOD


def main() -> None:
    sums = TotientSums(max(DEFAULT_N, 100_000))
    assert titanic_sets(1, sums) == 11
    assert titanic_sets(2, sums) == 494
    assert titanic_sets(4, sums) == 33_554_178
    assert titanic_sets(10, sums) == 60_631_646
    assert titanic_sets(20, sums) == 74_363_930
    assert titanic_sets(111, sums) == 13_500_401
    assert titanic_sets(100_000, sums) == 63_259_062

    print(titanic_sets(DEFAULT_N, sums))


if __name__ == "__main__":
    main()
