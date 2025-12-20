#!/usr/bin/env python3
"""
Project Euler 705: Total Inversion Count of Divided Sequences

We build the digit stream G(N) by iterating over primes < N, concatenating their decimal
representations while skipping any '0' digits.

For each digit d (1..9), we replace it by one of its divisors (uniformly among divisors),
forming all "divided sequences". The inversion count of a sequence equals the number of
pairs (i<j) with x_i > x_j. By linearity of expectation, the total over all divided
sequences equals:

    F(N) = (# divided sequences) * E[inversion count]

All computations are performed modulo 1_000_000_007.
"""

import math

MOD = 1_000_000_007

# Number of divisors for each digit 1..9 (0 unused)
W = [0, 1, 2, 2, 3, 2, 4, 2, 4, 3]

INV2 = pow(2, MOD - 2, MOD)
INV3 = pow(3, MOD - 2, MOD)
INV4 = pow(4, MOD - 2, MOD)


def _build_chunk_digits():
    """
    Precompute the non-zero decimal digits (in order) for every integer 0..9999.

    We use base-10000 chunks to avoid converting every prime to a string. Since zeros are
    ignored in G(N), leading zeros inside a chunk can also be ignored.
    """
    table = [()] * 10000
    out = [None] * 10000
    out[0] = ()
    for i in range(1, 10000):
        x = i
        ds = []
        while x:
            d = x % 10
            if d:
                ds.append(d)
            x //= 10
        ds.reverse()
        out[i] = tuple(ds)
    return out


CHUNK_DIGITS = _build_chunk_digits()


def _base_primes_upto(limit: int):
    """Primes <= limit using an odd-only sieve (fast for limit ~ 1e4)."""
    if limit < 2:
        return []
    # sieve[i] corresponds to odd number (2*i+1)
    sieve = bytearray((limit // 2) + 1)
    r = int(limit**0.5)
    for i in range(1, (r // 2) + 1):
        if sieve[i] == 0:
            p = 2 * i + 1
            start = (p * p) // 2
            sieve[start::p] = b"\x01" * (((len(sieve) - start - 1) // p) + 1)

    primes = [2]
    for i in range(1, len(sieve)):
        if sieve[i] == 0:
            p = 2 * i + 1
            if p <= limit:
                primes.append(p)
    return primes


def _primes_below(n: int, seg_odds: int = 1 << 20):
    """
    Yield all primes < n using a segmented odd-only sieve.

    seg_odds is the number of odd integers represented in each segment.
    """
    if n <= 2:
        return
    yield 2

    base = _base_primes_upto(int(math.isqrt(n - 1)))
    base = [
        p for p in base if p != 2
    ]  # only odd primes are needed for odd-only segments

    step = 2 * seg_odds
    low = 3  # odd

    while low < n:
        high = min(low + step, n)
        # number of odd integers in [low, high)
        size = ((high - low) + 1) // 2
        sieve = bytearray(size)  # 0 = prime candidate, 1 = composite

        for p in base:
            pp = p * p
            if pp >= high:
                break
            start = pp if pp > low else ((low + p - 1) // p) * p
            if start % 2 == 0:
                start += p
            idx = (start - low) // 2
            if idx < size:
                # In odd-only indexing, stepping by 2p in values equals stepping by p in indices.
                sieve[idx::p] = b"\x01" * (((size - idx - 1) // p) + 1)

        for i, is_comp in enumerate(sieve):
            if is_comp == 0:
                yield low + 2 * i

        low += step


def compute_F(N: int) -> int:
    """
    Compute F(N) modulo MOD.

    Let each position choose uniformly among divisors of its master digit.
    Maintain:
      - A[v] = sum_{positions so far} P(chosen digit == v)  for v=1..9
      - E    = expected inversion count accumulated so far (mod MOD)
      - nmod = number of processed digits so far (mod MOD)

    Then for a new position with distribution P_j:
      contribution = sum_t P_j(t) * sum_{u>t} A[u]
    and we add P_j into A afterwards.
    """
    if N <= 2:
        return 0

    counts = [
        0
    ] * 10  # counts of original (master) digits 1..9, with zeros already ignored

    # A1..A9 are A[1]..A[9] in mod space
    A1 = A2 = A3 = A4 = A5 = A6 = A7 = A8 = A9 = 0

    E = 0
    nmod = 0  # number of processed digits so far, modulo MOD

    for p in _primes_below(N):
        q, r = divmod(p, 10000)
        # Process high chunk, then low chunk (both ignore zeros via CHUNK_DIGITS)
        for d in CHUNK_DIGITS[q]:
            # ---- process digit d ----
            counts[d] += 1
            s1 = A1  # prefix sum up to 1

            if d == 1:
                t1 = nmod - s1
                if t1 < 0:
                    t1 += MOD
                E += t1
                if E >= MOD:
                    E -= MOD
                A1 = s1 + 1
                if A1 >= MOD:
                    A1 -= MOD

            elif d == 2:
                inv = INV2
                s2 = s1 + A2
                if s2 >= MOD:
                    s2 -= MOD
                t1 = nmod - s1
                if t1 < 0:
                    t1 += MOD
                t2 = nmod - s2
                if t2 < 0:
                    t2 += MOD
                term = t1 + t2
                if term >= MOD:
                    term -= MOD
                add = (inv * term) % MOD
                E += add
                if E >= MOD:
                    E -= MOD
                A1 = s1 + inv
                if A1 >= MOD:
                    A1 -= MOD
                A2 += inv
                if A2 >= MOD:
                    A2 -= MOD

            elif d == 3:
                inv = INV2
                s2 = s1 + A2
                if s2 >= MOD:
                    s2 -= MOD
                s3 = s2 + A3
                if s3 >= MOD:
                    s3 -= MOD
                t1 = nmod - s1
                if t1 < 0:
                    t1 += MOD
                t3 = nmod - s3
                if t3 < 0:
                    t3 += MOD
                term = t1 + t3
                if term >= MOD:
                    term -= MOD
                add = (inv * term) % MOD
                E += add
                if E >= MOD:
                    E -= MOD
                A1 = s1 + inv
                if A1 >= MOD:
                    A1 -= MOD
                A3 += inv
                if A3 >= MOD:
                    A3 -= MOD

            elif d == 4:
                inv = INV3
                s2 = s1 + A2
                if s2 >= MOD:
                    s2 -= MOD
                s3 = s2 + A3
                if s3 >= MOD:
                    s3 -= MOD
                s4 = s3 + A4
                if s4 >= MOD:
                    s4 -= MOD
                t1 = nmod - s1
                if t1 < 0:
                    t1 += MOD
                t2 = nmod - s2
                if t2 < 0:
                    t2 += MOD
                t4 = nmod - s4
                if t4 < 0:
                    t4 += MOD
                term = t1 + t2
                if term >= MOD:
                    term -= MOD
                term += t4
                if term >= MOD:
                    term -= MOD
                add = (inv * term) % MOD
                E += add
                if E >= MOD:
                    E -= MOD
                A1 = s1 + inv
                if A1 >= MOD:
                    A1 -= MOD
                A2 += inv
                if A2 >= MOD:
                    A2 -= MOD
                A4 += inv
                if A4 >= MOD:
                    A4 -= MOD

            elif d == 5:
                inv = INV2
                s2 = s1 + A2
                if s2 >= MOD:
                    s2 -= MOD
                s3 = s2 + A3
                if s3 >= MOD:
                    s3 -= MOD
                s4 = s3 + A4
                if s4 >= MOD:
                    s4 -= MOD
                s5 = s4 + A5
                if s5 >= MOD:
                    s5 -= MOD
                t1 = nmod - s1
                if t1 < 0:
                    t1 += MOD
                t5 = nmod - s5
                if t5 < 0:
                    t5 += MOD
                term = t1 + t5
                if term >= MOD:
                    term -= MOD
                add = (inv * term) % MOD
                E += add
                if E >= MOD:
                    E -= MOD
                A1 = s1 + inv
                if A1 >= MOD:
                    A1 -= MOD
                A5 += inv
                if A5 >= MOD:
                    A5 -= MOD

            elif d == 6:
                inv = INV4
                s2 = s1 + A2
                if s2 >= MOD:
                    s2 -= MOD
                s3 = s2 + A3
                if s3 >= MOD:
                    s3 -= MOD
                s4 = s3 + A4
                if s4 >= MOD:
                    s4 -= MOD
                s5 = s4 + A5
                if s5 >= MOD:
                    s5 -= MOD
                s6 = s5 + A6
                if s6 >= MOD:
                    s6 -= MOD
                t1 = nmod - s1
                if t1 < 0:
                    t1 += MOD
                t2 = nmod - s2
                if t2 < 0:
                    t2 += MOD
                t3 = nmod - s3
                if t3 < 0:
                    t3 += MOD
                t6 = nmod - s6
                if t6 < 0:
                    t6 += MOD
                term = t1 + t2
                if term >= MOD:
                    term -= MOD
                term += t3
                if term >= MOD:
                    term -= MOD
                term += t6
                if term >= MOD:
                    term -= MOD
                add = (inv * term) % MOD
                E += add
                if E >= MOD:
                    E -= MOD
                A1 = s1 + inv
                if A1 >= MOD:
                    A1 -= MOD
                A2 += inv
                if A2 >= MOD:
                    A2 -= MOD
                A3 += inv
                if A3 >= MOD:
                    A3 -= MOD
                A6 += inv
                if A6 >= MOD:
                    A6 -= MOD

            elif d == 7:
                inv = INV2
                s2 = s1 + A2
                if s2 >= MOD:
                    s2 -= MOD
                s3 = s2 + A3
                if s3 >= MOD:
                    s3 -= MOD
                s4 = s3 + A4
                if s4 >= MOD:
                    s4 -= MOD
                s5 = s4 + A5
                if s5 >= MOD:
                    s5 -= MOD
                s6 = s5 + A6
                if s6 >= MOD:
                    s6 -= MOD
                s7 = s6 + A7
                if s7 >= MOD:
                    s7 -= MOD
                t1 = nmod - s1
                if t1 < 0:
                    t1 += MOD
                t7 = nmod - s7
                if t7 < 0:
                    t7 += MOD
                term = t1 + t7
                if term >= MOD:
                    term -= MOD
                add = (inv * term) % MOD
                E += add
                if E >= MOD:
                    E -= MOD
                A1 = s1 + inv
                if A1 >= MOD:
                    A1 -= MOD
                A7 += inv
                if A7 >= MOD:
                    A7 -= MOD

            elif d == 8:
                inv = INV4
                s2 = s1 + A2
                if s2 >= MOD:
                    s2 -= MOD
                s3 = s2 + A3
                if s3 >= MOD:
                    s3 -= MOD
                s4 = s3 + A4
                if s4 >= MOD:
                    s4 -= MOD
                s5 = s4 + A5
                if s5 >= MOD:
                    s5 -= MOD
                s6 = s5 + A6
                if s6 >= MOD:
                    s6 -= MOD
                s7 = s6 + A7
                if s7 >= MOD:
                    s7 -= MOD
                s8 = s7 + A8
                if s8 >= MOD:
                    s8 -= MOD
                t1 = nmod - s1
                if t1 < 0:
                    t1 += MOD
                t2 = nmod - s2
                if t2 < 0:
                    t2 += MOD
                t4 = nmod - s4
                if t4 < 0:
                    t4 += MOD
                t8 = nmod - s8
                if t8 < 0:
                    t8 += MOD
                term = t1 + t2
                if term >= MOD:
                    term -= MOD
                term += t4
                if term >= MOD:
                    term -= MOD
                term += t8
                if term >= MOD:
                    term -= MOD
                add = (inv * term) % MOD
                E += add
                if E >= MOD:
                    E -= MOD
                A1 = s1 + inv
                if A1 >= MOD:
                    A1 -= MOD
                A2 += inv
                if A2 >= MOD:
                    A2 -= MOD
                A4 += inv
                if A4 >= MOD:
                    A4 -= MOD
                A8 += inv
                if A8 >= MOD:
                    A8 -= MOD

            else:  # d == 9
                inv = INV3
                s2 = s1 + A2
                if s2 >= MOD:
                    s2 -= MOD
                s3 = s2 + A3
                if s3 >= MOD:
                    s3 -= MOD
                t1 = nmod - s1
                if t1 < 0:
                    t1 += MOD
                t3 = nmod - s3
                if t3 < 0:
                    t3 += MOD
                term = t1 + t3
                if term >= MOD:
                    term -= MOD
                add = (inv * term) % MOD
                E += add
                if E >= MOD:
                    E -= MOD
                A1 = s1 + inv
                if A1 >= MOD:
                    A1 -= MOD
                A3 += inv
                if A3 >= MOD:
                    A3 -= MOD

            nmod += 1
            if nmod == MOD:
                nmod = 0

        for d in CHUNK_DIGITS[r]:
            # ---- process digit d ----
            counts[d] += 1
            s1 = A1

            if d == 1:
                t1 = nmod - s1
                if t1 < 0:
                    t1 += MOD
                E += t1
                if E >= MOD:
                    E -= MOD
                A1 = s1 + 1
                if A1 >= MOD:
                    A1 -= MOD

            elif d == 2:
                inv = INV2
                s2 = s1 + A2
                if s2 >= MOD:
                    s2 -= MOD
                t1 = nmod - s1
                if t1 < 0:
                    t1 += MOD
                t2 = nmod - s2
                if t2 < 0:
                    t2 += MOD
                term = t1 + t2
                if term >= MOD:
                    term -= MOD
                add = (inv * term) % MOD
                E += add
                if E >= MOD:
                    E -= MOD
                A1 = s1 + inv
                if A1 >= MOD:
                    A1 -= MOD
                A2 += inv
                if A2 >= MOD:
                    A2 -= MOD

            elif d == 3:
                inv = INV2
                s2 = s1 + A2
                if s2 >= MOD:
                    s2 -= MOD
                s3 = s2 + A3
                if s3 >= MOD:
                    s3 -= MOD
                t1 = nmod - s1
                if t1 < 0:
                    t1 += MOD
                t3 = nmod - s3
                if t3 < 0:
                    t3 += MOD
                term = t1 + t3
                if term >= MOD:
                    term -= MOD
                add = (inv * term) % MOD
                E += add
                if E >= MOD:
                    E -= MOD
                A1 = s1 + inv
                if A1 >= MOD:
                    A1 -= MOD
                A3 += inv
                if A3 >= MOD:
                    A3 -= MOD

            elif d == 4:
                inv = INV3
                s2 = s1 + A2
                if s2 >= MOD:
                    s2 -= MOD
                s3 = s2 + A3
                if s3 >= MOD:
                    s3 -= MOD
                s4 = s3 + A4
                if s4 >= MOD:
                    s4 -= MOD
                t1 = nmod - s1
                if t1 < 0:
                    t1 += MOD
                t2 = nmod - s2
                if t2 < 0:
                    t2 += MOD
                t4 = nmod - s4
                if t4 < 0:
                    t4 += MOD
                term = t1 + t2
                if term >= MOD:
                    term -= MOD
                term += t4
                if term >= MOD:
                    term -= MOD
                add = (inv * term) % MOD
                E += add
                if E >= MOD:
                    E -= MOD
                A1 = s1 + inv
                if A1 >= MOD:
                    A1 -= MOD
                A2 += inv
                if A2 >= MOD:
                    A2 -= MOD
                A4 += inv
                if A4 >= MOD:
                    A4 -= MOD

            elif d == 5:
                inv = INV2
                s2 = s1 + A2
                if s2 >= MOD:
                    s2 -= MOD
                s3 = s2 + A3
                if s3 >= MOD:
                    s3 -= MOD
                s4 = s3 + A4
                if s4 >= MOD:
                    s4 -= MOD
                s5 = s4 + A5
                if s5 >= MOD:
                    s5 -= MOD
                t1 = nmod - s1
                if t1 < 0:
                    t1 += MOD
                t5 = nmod - s5
                if t5 < 0:
                    t5 += MOD
                term = t1 + t5
                if term >= MOD:
                    term -= MOD
                add = (inv * term) % MOD
                E += add
                if E >= MOD:
                    E -= MOD
                A1 = s1 + inv
                if A1 >= MOD:
                    A1 -= MOD
                A5 += inv
                if A5 >= MOD:
                    A5 -= MOD

            elif d == 6:
                inv = INV4
                s2 = s1 + A2
                if s2 >= MOD:
                    s2 -= MOD
                s3 = s2 + A3
                if s3 >= MOD:
                    s3 -= MOD
                s4 = s3 + A4
                if s4 >= MOD:
                    s4 -= MOD
                s5 = s4 + A5
                if s5 >= MOD:
                    s5 -= MOD
                s6 = s5 + A6
                if s6 >= MOD:
                    s6 -= MOD
                t1 = nmod - s1
                if t1 < 0:
                    t1 += MOD
                t2 = nmod - s2
                if t2 < 0:
                    t2 += MOD
                t3 = nmod - s3
                if t3 < 0:
                    t3 += MOD
                t6 = nmod - s6
                if t6 < 0:
                    t6 += MOD
                term = t1 + t2
                if term >= MOD:
                    term -= MOD
                term += t3
                if term >= MOD:
                    term -= MOD
                term += t6
                if term >= MOD:
                    term -= MOD
                add = (inv * term) % MOD
                E += add
                if E >= MOD:
                    E -= MOD
                A1 = s1 + inv
                if A1 >= MOD:
                    A1 -= MOD
                A2 += inv
                if A2 >= MOD:
                    A2 -= MOD
                A3 += inv
                if A3 >= MOD:
                    A3 -= MOD
                A6 += inv
                if A6 >= MOD:
                    A6 -= MOD

            elif d == 7:
                inv = INV2
                s2 = s1 + A2
                if s2 >= MOD:
                    s2 -= MOD
                s3 = s2 + A3
                if s3 >= MOD:
                    s3 -= MOD
                s4 = s3 + A4
                if s4 >= MOD:
                    s4 -= MOD
                s5 = s4 + A5
                if s5 >= MOD:
                    s5 -= MOD
                s6 = s5 + A6
                if s6 >= MOD:
                    s6 -= MOD
                s7 = s6 + A7
                if s7 >= MOD:
                    s7 -= MOD
                t1 = nmod - s1
                if t1 < 0:
                    t1 += MOD
                t7 = nmod - s7
                if t7 < 0:
                    t7 += MOD
                term = t1 + t7
                if term >= MOD:
                    term -= MOD
                add = (inv * term) % MOD
                E += add
                if E >= MOD:
                    E -= MOD
                A1 = s1 + inv
                if A1 >= MOD:
                    A1 -= MOD
                A7 += inv
                if A7 >= MOD:
                    A7 -= MOD

            elif d == 8:
                inv = INV4
                s2 = s1 + A2
                if s2 >= MOD:
                    s2 -= MOD
                s3 = s2 + A3
                if s3 >= MOD:
                    s3 -= MOD
                s4 = s3 + A4
                if s4 >= MOD:
                    s4 -= MOD
                s5 = s4 + A5
                if s5 >= MOD:
                    s5 -= MOD
                s6 = s5 + A6
                if s6 >= MOD:
                    s6 -= MOD
                s7 = s6 + A7
                if s7 >= MOD:
                    s7 -= MOD
                s8 = s7 + A8
                if s8 >= MOD:
                    s8 -= MOD
                t1 = nmod - s1
                if t1 < 0:
                    t1 += MOD
                t2 = nmod - s2
                if t2 < 0:
                    t2 += MOD
                t4 = nmod - s4
                if t4 < 0:
                    t4 += MOD
                t8 = nmod - s8
                if t8 < 0:
                    t8 += MOD
                term = t1 + t2
                if term >= MOD:
                    term -= MOD
                term += t4
                if term >= MOD:
                    term -= MOD
                term += t8
                if term >= MOD:
                    term -= MOD
                add = (inv * term) % MOD
                E += add
                if E >= MOD:
                    E -= MOD
                A1 = s1 + inv
                if A1 >= MOD:
                    A1 -= MOD
                A2 += inv
                if A2 >= MOD:
                    A2 -= MOD
                A4 += inv
                if A4 >= MOD:
                    A4 -= MOD
                A8 += inv
                if A8 >= MOD:
                    A8 -= MOD

            else:  # d == 9
                inv = INV3
                s2 = s1 + A2
                if s2 >= MOD:
                    s2 -= MOD
                s3 = s2 + A3
                if s3 >= MOD:
                    s3 -= MOD
                t1 = nmod - s1
                if t1 < 0:
                    t1 += MOD
                t3 = nmod - s3
                if t3 < 0:
                    t3 += MOD
                term = t1 + t3
                if term >= MOD:
                    term -= MOD
                add = (inv * term) % MOD
                E += add
                if E >= MOD:
                    E -= MOD
                A1 = s1 + inv
                if A1 >= MOD:
                    A1 -= MOD
                A3 += inv
                if A3 >= MOD:
                    A3 -= MOD

            nmod += 1
            if nmod == MOD:
                nmod = 0

    # Number of divided sequences: M = product over positions of W[d_position].
    # Compute via digit counts: M = Π_d W[d]^(count[d]).
    M = 1
    for d in range(1, 10):
        c = counts[d]
        if c:
            M = (M * pow(W[d], c, MOD)) % MOD

    return (M * E) % MOD


def main():
    # Test values from the problem statement
    assert compute_F(20) == 3312
    assert compute_F(50) == 338079744

    print(compute_F(100_000_000))


if __name__ == "__main__":
    main()
