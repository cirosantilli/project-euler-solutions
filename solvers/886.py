#!/usr/bin/env python3
"""
Project Euler 886: Coprime Permutations

Count permutations of {2,3,...,n} such that adjacent numbers are coprime.
For n=34, output P(34) mod 83,456,729.

Key observations used:
- All even numbers are pairwise non-coprime (share factor 2), so evens cannot be adjacent.
- Therefore any valid permutation must alternate parity.
- Reduce cross-parity coprimality checks to bitmasks of *relevant odd primes*.
- Compress numbers into "types" by their relevant-prime masks.
- Count alternating type-sequences via DP with memoization over remaining type-counts.
- Multiply by factorials within each type to restore distinct labels.

No external libraries used.
"""

import sys


MOD = 83_456_729


def sieve_primes_upto(n: int) -> list[int]:
    if n < 2:
        return []
    sieve = bytearray(b"\x01") * (n + 1)
    sieve[0:2] = b"\x00\x00"
    p = 2
    while p * p <= n:
        if sieve[p]:
            step = p
            start = p * p
            sieve[start : n + 1 : step] = b"\x00" * (((n - start) // step) + 1)
        p += 1
    return [i for i in range(2, n + 1) if sieve[i]]


def odd_prime_factors_set(x: int, primes: list[int]) -> set[int]:
    """Return set of odd prime factors of x (removing all powers of 2 first)."""
    while (x & 1) == 0:
        x //= 2
    s: set[int] = set()
    for p in primes:
        if p == 2:
            continue
        if p * p > x:
            break
        if x % p == 0:
            s.add(p)
            while x % p == 0:
                x //= p
    if x > 1:  # remaining odd prime
        s.add(x)
    return s


def build_mixed_radix_tables(max_counts: list[int]):
    """
    Build:
      - multipliers for encoding a vector of remaining counts into an int index
      - dec[idx][i] = idx' if we decrement count i (or -1 if not possible)
      - sum_counts[idx] = sum of digits (remaining total items in that side)
    """
    rad = [c + 1 for c in max_counts]
    mult = [1] * len(max_counts)
    prod = 1
    for i in range(len(max_counts)):
        mult[i] = prod
        prod *= rad[i]
    total_states = prod

    # Precompute decrement transitions and sums
    dec = [[-1] * len(max_counts) for _ in range(total_states)]
    sum_counts = [0] * total_states
    for idx in range(total_states):
        s = 0
        for i in range(len(max_counts)):
            d = (idx // mult[i]) % rad[i]
            s += d
            if d:
                dec[idx][i] = idx - mult[i]
        sum_counts[idx] = s

    init_idx = 0
    for c, m in zip(max_counts, mult):
        init_idx += c * m

    return total_states, mult, dec, sum_counts, init_idx


def factorials_mod(n: int, mod: int) -> list[int]:
    f = [1] * (n + 1)
    for i in range(2, n + 1):
        f[i] = (f[i - 1] * i) % mod
    return f


def count_alternating_sequences(
    even_masks: list[int],
    even_counts: list[int],
    odd_masks: list[int],
    odd_counts: list[int],
    mod: int,
    start_with_even: bool,
) -> int:
    """
    Count alternating sequences of type-masks using all items from both sides exactly once:
      if start_with_even:
        E O E O ... (end parity determined by counts)
      else:
        O E O E ...
    Items within each mask-type are indistinguishable here.
    """
    # Build mixed-radix encodings for remaining-count vectors
    odd_total, _, odd_dec, odd_sum, odd_init = build_mixed_radix_tables(odd_counts)
    even_total, _, even_dec, even_sum, even_init = build_mixed_radix_tables(even_counts)

    # Compatibility lists: type i on one side can follow type j on the other iff masks disjoint
    even_to_odd = [[] for _ in range(len(even_masks))]
    odd_to_even = [[] for _ in range(len(odd_masks))]
    for i, em in enumerate(even_masks):
        for j, om in enumerate(odd_masks):
            if (em & om) == 0:
                even_to_odd[i].append(j)
                odd_to_even[j].append(i)

    # Memoization dictionaries keyed by packed integers
    memo_even: dict[int, int] = {}
    memo_odd: dict[int, int] = {}

    # Pack keys tightly to reduce tuple overhead
    # For dp_even: key = ((e_idx * odd_total + o_idx) * len(even_masks) + last_even_type)
    # For dp_odd:  key = ((e_idx * odd_total + o_idx) * len(odd_masks)  + last_odd_type)
    def dp_even(last_i: int, e_idx: int, o_idx: int) -> int:
        if even_sum[e_idx] == 0 and odd_sum[o_idx] == 0:
            return 1
        if odd_sum[o_idx] == 0:
            return 0
        key = (e_idx * odd_total + o_idx) * len(even_masks) + last_i
        v = memo_even.get(key)
        if v is not None:
            return v

        total = 0
        for j in even_to_odd[last_i]:
            no = odd_dec[o_idx][j]
            if no != -1:
                total += dp_odd(j, e_idx, no)
        total %= mod
        memo_even[key] = total
        return total

    def dp_odd(last_j: int, e_idx: int, o_idx: int) -> int:
        if even_sum[e_idx] == 0 and odd_sum[o_idx] == 0:
            return 1
        if even_sum[e_idx] == 0:
            return 0
        key = (e_idx * odd_total + o_idx) * len(odd_masks) + last_j
        v = memo_odd.get(key)
        if v is not None:
            return v

        total = 0
        for i in odd_to_even[last_j]:
            ne = even_dec[e_idx][i]
            if ne != -1:
                total += dp_even(i, ne, o_idx)
        total %= mod
        memo_odd[key] = total
        return total

    total = 0
    if start_with_even:
        # First choose an even type, then alternate
        for i in range(len(even_masks)):
            ne = even_dec[even_init][i]
            if ne != -1:
                total += dp_even(i, ne, odd_init)
    else:
        # First choose an odd type, then alternate
        for j in range(len(odd_masks)):
            no = odd_dec[odd_init][j]
            if no != -1:
                total += dp_odd(j, even_init, no)

    return total % mod


def P(n: int, mod: int = MOD) -> int:
    """
    Return P(n) modulo mod.
    """
    if n < 2:
        return 0

    nums = list(range(2, n + 1))
    evens = [x for x in nums if (x & 1) == 0]
    odds = [x for x in nums if (x & 1) == 1]

    # Parity alternation feasibility
    if abs(len(evens) - len(odds)) > 1:
        return 0

    primes = sieve_primes_upto(n)

    # Only odd primes that appear in *evens* can ever block coprimality across parity.
    relevant_primes_set: set[int] = set()
    for e in evens:
        relevant_primes_set |= odd_prime_factors_set(e, primes)

    relevant_primes = sorted(relevant_primes_set)
    prime_to_bit = {p: i for i, p in enumerate(relevant_primes)}

    def mask_of(x: int) -> int:
        # mask over relevant odd primes only
        m = 0
        for p in relevant_primes:
            if x % p == 0:
                m |= 1 << prime_to_bit[p]
        return m

    # Compress each side by mask
    even_count_map: dict[int, int] = {}
    odd_count_map: dict[int, int] = {}
    for e in evens:
        m = mask_of(e)
        even_count_map[m] = even_count_map.get(m, 0) + 1
    for o in odds:
        m = mask_of(o)
        odd_count_map[m] = odd_count_map.get(m, 0) + 1

    even_masks = sorted(even_count_map.keys())
    odd_masks = sorted(odd_count_map.keys())
    even_counts = [even_count_map[m] for m in even_masks]
    odd_counts = [odd_count_map[m] for m in odd_masks]

    # Count alternating type-sequences (indistinguishable within each type)
    seq_count = 0
    if len(evens) == len(odds) + 1:
        # Must start/end with even
        seq_count = count_alternating_sequences(
            even_masks, even_counts, odd_masks, odd_counts, mod, start_with_even=True
        )
    elif len(odds) == len(evens) + 1:
        # Must start/end with odd
        seq_count = count_alternating_sequences(
            even_masks, even_counts, odd_masks, odd_counts, mod, start_with_even=False
        )
    else:
        # Equal sizes: can start with either parity
        a = count_alternating_sequences(
            even_masks, even_counts, odd_masks, odd_counts, mod, start_with_even=True
        )
        b = count_alternating_sequences(
            even_masks, even_counts, odd_masks, odd_counts, mod, start_with_even=False
        )
        seq_count = (a + b) % mod

    # Multiply by factorials within each mask-type to distinguish labels
    facts = factorials_mod(n, mod)
    ways_with_labels = seq_count
    for c in even_counts:
        ways_with_labels = (ways_with_labels * facts[c]) % mod
    for c in odd_counts:
        ways_with_labels = (ways_with_labels * facts[c]) % mod

    return ways_with_labels


def main() -> None:
    # Test values from the problem statement
    assert P(4, MOD) == 2
    assert P(10, MOD) == 576

    print(P(34, MOD))


if __name__ == "__main__":
    sys.setrecursionlimit(1_000_000)
    main()
