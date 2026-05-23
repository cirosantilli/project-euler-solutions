#!/usr/bin/env python
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
DIVISORS = (
    (),
    (1,),
    (1, 2),
    (1, 3),
    (1, 2, 4),
    (1, 5),
    (1, 2, 3, 6),
    (1, 7),
    (1, 2, 4, 8),
    (1, 3, 9),
)


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


def _build_chunk_transforms():
    """
    Precompute the linear DP transform for each base-10000 digit chunk.

    The direct DP state is:
      T       = number of divided prefixes
      I       = total inversion count over those prefixes
      A[v]    = total occurrences of value v over all prefixes

    A whole chunk transforms the state as:
      T'    = m*T
      A'[v] = m*A[v] + b[v]*T
      I'    = m*I + sum_v c[v]*A[v] + e*T
    """
    transforms = []

    for digits in CHUNK_DIGITS:
        m = 1
        b = [0] * 10
        c = [0] * 10
        e = 0

        for d in digits:
            w = W[d]
            divisors = DIVISORS[d]

            suffix_from_b = 0
            less_count = [0] * 10
            for v in divisors:
                for u in range(v + 1, 10):
                    suffix_from_b += b[u]
                    less_count[u] += 1

            old_m = m
            m = (m * w) % MOD
            for u in range(1, 10):
                c[u] = (w * c[u] + old_m * less_count[u]) % MOD
                b[u] = (w * b[u] + (old_m if u in divisors else 0)) % MOD
            e = (w * e + suffix_from_b) % MOD

        transforms.append(
            (
                m,
                b[1],
                b[2],
                b[3],
                b[4],
                b[5],
                b[6],
                b[7],
                b[8],
                b[9],
                c[1],
                c[2],
                c[3],
                c[4],
                c[5],
                c[6],
                c[7],
                c[8],
                c[9],
                e,
            )
        )

    return transforms


CHUNK_TRANSFORMS = _build_chunk_transforms()


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

    The state tracks totals over all divided prefixes directly.  Each base-10000
    chunk has a precomputed linear transform, so every prime contributes at most
    two compact state updates instead of one update per decimal digit.
    """
    if N <= 2:
        return 0

    T = 1
    I = 0
    A1 = A2 = A3 = A4 = A5 = A6 = A7 = A8 = A9 = 0
    transforms = CHUNK_TRANSFORMS

    for p in _primes_below(N):
        q, r = divmod(p, 10000)

        if q:
            (
                m,
                b1,
                b2,
                b3,
                b4,
                b5,
                b6,
                b7,
                b8,
                b9,
                c1,
                c2,
                c3,
                c4,
                c5,
                c6,
                c7,
                c8,
                c9,
                e,
            ) = transforms[q]
            I = (
                m * I
                + c1 * A1
                + c2 * A2
                + c3 * A3
                + c4 * A4
                + c5 * A5
                + c6 * A6
                + c7 * A7
                + c8 * A8
                + c9 * A9
                + e * T
            ) % MOD
            A1 = (m * A1 + b1 * T) % MOD
            A2 = (m * A2 + b2 * T) % MOD
            A3 = (m * A3 + b3 * T) % MOD
            A4 = (m * A4 + b4 * T) % MOD
            A5 = (m * A5 + b5 * T) % MOD
            A6 = (m * A6 + b6 * T) % MOD
            A7 = (m * A7 + b7 * T) % MOD
            A8 = (m * A8 + b8 * T) % MOD
            A9 = (m * A9 + b9 * T) % MOD
            T = (m * T) % MOD

        (
            m,
            b1,
            b2,
            b3,
            b4,
            b5,
            b6,
            b7,
            b8,
            b9,
            c1,
            c2,
            c3,
            c4,
            c5,
            c6,
            c7,
            c8,
            c9,
            e,
        ) = transforms[r]
        I = (
            m * I
            + c1 * A1
            + c2 * A2
            + c3 * A3
            + c4 * A4
            + c5 * A5
            + c6 * A6
            + c7 * A7
            + c8 * A8
            + c9 * A9
            + e * T
        ) % MOD
        A1 = (m * A1 + b1 * T) % MOD
        A2 = (m * A2 + b2 * T) % MOD
        A3 = (m * A3 + b3 * T) % MOD
        A4 = (m * A4 + b4 * T) % MOD
        A5 = (m * A5 + b5 * T) % MOD
        A6 = (m * A6 + b6 * T) % MOD
        A7 = (m * A7 + b7 * T) % MOD
        A8 = (m * A8 + b8 * T) % MOD
        A9 = (m * A9 + b9 * T) % MOD
        T = (m * T) % MOD

    return I


def main():
    # Test values from the problem statement
    assert compute_F(20) == 3312
    assert compute_F(50) == 338079744

    print(compute_F(100_000_000))


if __name__ == "__main__":
    main()
