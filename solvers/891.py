#!/usr/bin/env python3
"""
Project Euler 891: Ambiguous Clock

A clock has three identical hands (hour, minute, second) moving continuously.
There is no reference mark, so the whole clock may be rotated by an unknown
amount; and the hands are unlabeled. We count all moments within a 12-hour
cycle where the displayed configuration is consistent with at least two
different times.

Implementation notes
- Work in a 12-hour cycle length L = 43200 seconds.
- Hand angular speeds (in revolutions per 12-hour cycle) are integers:
    hour:   1
    minute: 12
    second: 720
- Two times t and t' are indistinguishable iff there exists a permutation of
  the hands and a global rotation making the three angles coincide.
- Eliminating the unknown rotation yields two independent linear congruences
  (using differences between one chosen hand and the other two).
- For each non-identity hand permutation, we enumerate the small bounded
  integer "wrap counts" and solve the resulting 2x2 system exactly.
- Times are stored as reduced rationals (numerator/denominator) packed into a
  single integer key for fast deduplication.

No external libraries are used (standard library only).
"""

from itertools import permutations
from math import gcd

# One 12-hour cycle in seconds.
L = 12 * 60 * 60  # 43200

# Hand speeds measured in "revolutions per 12-hour cycle".
# If time is t seconds, the hand angles (in turns) are v[i] * t / L (mod 1).
V = (1, 12, 720)  # hour, minute, second

# Packing for (numerator, denominator) reduced fraction representing seconds.
# Denominators in this problem stay below 1<<20, so we pack into one integer.
_SHIFT = 1 << 20


def _ceil_div(n: int, d: int) -> int:
    """ceil(n/d) for d>0, using integer arithmetic."""
    return -((-n) // d)


def _k2_range(u: int, v: int, d: int) -> tuple[int, int]:
    """
    Solve 0 <= u - v*k < d for integer k, returning inclusive [lo, hi].
    Here d > 0 and v != 0.
    """
    if v > 0:
        lo = _ceil_div(u - d + 1, v)
        hi = u // v
    else:
        vp = -v
        lo = _ceil_div(-u, vp)
        hi = (d - 1 - u) // vp
    return lo, hi


def _add_time_key(s: set[int], l1: int, d1: int, num: int) -> None:
    """
    Add time t = (l1 * num) / d1 (seconds) to the set, reduced.
    Here gcd(l1, d1) == 1 and 0 <= num < d1*something; we reduce by gcd(num,d1).
    """
    g = gcd(num, d1)
    n = l1 * (num // g)
    den = d1 // g
    s.add(n * _SHIFT + den)


def ambiguous_moments_count() -> int:
    """
    Return the number of ambiguous moments in one 12-hour cycle.

    The method builds the set of all times that participate in at least one
    nontrivial equivalence with another time (under a hand permutation and
    global rotation).
    """
    ambiguous: set[int] = set()
    add = ambiguous.add
    gcd_local = gcd

    # Use differences relative to the "hour" slot at time t.
    # Constants: a = V[0]-V[1], c = V[0]-V[2]
    a = V[0] - V[1]  # -11
    c = V[0] - V[2]  # -719

    for perm in permutations(range(3)):
        if perm == (0, 1, 2):
            continue  # identity gives only t == t' within one cycle

        s0, s1, s2 = perm

        # Build the 2x2 system from matching pairwise differences.
        b = -(V[s0] - V[s1])
        d = -(V[s0] - V[s2])

        det = a * d - c * b
        sign = 1
        if det < 0:
            sign = -1
            det = -det
        D = det  # positive

        # Wrap-count bounds: since 0<=t,t'<L, these integers are small.
        K1 = abs(a) + abs(b)
        K2 = abs(c) + abs(d)

        # Reduce (L * num)/D efficiently:
        g0 = gcd_local(L, D)
        l1 = L // g0
        d1 = D // g0  # gcd(l1, d1) == 1

        # For a fixed k1, both constraints 0 <= num_t < D and 0 <= num_t' < D
        # impose an interval of valid k2 values. Intersect them to avoid
        # scanning the full rectangle.
        for k1 in range(-K1, K1 + 1):
            u1 = sign * k1 * d
            v1 = sign * b

            # num_t' = sign*(a*k2 - c*k1) = sign*(719*k1 - 11*k2)
            u2 = sign * 719 * k1
            v2 = sign * 11

            lo1, hi1 = _k2_range(u1, v1, D)
            lo2, hi2 = _k2_range(u2, v2, D)

            lo = max(-K2, lo1, lo2)
            hi = min(K2, hi1, hi2)
            if lo > hi:
                continue

            for k2 in range(lo, hi + 1):
                num_t = u1 - v1 * k2
                num_tp = u2 - v2 * k2
                if num_t == num_tp:
                    continue  # would imply t == t'

                # Reduce and pack both times:
                g = gcd_local(num_t, d1)
                n = l1 * (num_t // g)
                den = d1 // g
                add(n * _SHIFT + den)

                g = gcd_local(num_tp, d1)
                n = l1 * (num_tp // g)
                den = d1 // g
                add(n * _SHIFT + den)

    return len(ambiguous)


def _key_for_integer_seconds(seconds: int) -> int:
    """Packed key for an integer-second time."""
    return seconds * _SHIFT + 1


def main() -> None:
    # Build the count (and implicitly the ambiguous set) once.
    # The problem statement provides these example moments:
    #
    # - 12:00:00 (all three hands coincide) is NOT ambiguous.
    # - 1:30:00 and 7:30:00 are ambiguous (rotation + hand swap).
    # - 3:00:00 and 9:00:00 are NOT ambiguous.
    #
    # We check these examples by reconstructing the ambiguous set during count.
    # To keep the implementation simple and exact, we recompute the set here.
    #
    # (The full count is printed; do not hard-code it.)
    #
    # NOTE: These asserts are intentionally light-weight relative to the full
    # computation; they validate interpretation of the statement.
    #
    ambiguous: set[int] = set()

    # Re-run the same generation as ambiguous_moments_count(), but keep the set.
    add = ambiguous.add
    gcd_local = gcd
    a = V[0] - V[1]  # -11
    c = V[0] - V[2]  # -719

    for perm in permutations(range(3)):
        if perm == (0, 1, 2):
            continue
        s0, s1, s2 = perm
        b = -(V[s0] - V[s1])
        d = -(V[s0] - V[s2])
        det = a * d - c * b
        sign = 1
        if det < 0:
            sign = -1
            det = -det
        D = det
        K1 = abs(a) + abs(b)
        K2 = abs(c) + abs(d)

        g0 = gcd_local(L, D)
        l1 = L // g0
        d1 = D // g0

        for k1 in range(-K1, K1 + 1):
            u1 = sign * k1 * d
            v1 = sign * b
            u2 = sign * 719 * k1
            v2 = sign * 11

            lo1, hi1 = _k2_range(u1, v1, D)
            lo2, hi2 = _k2_range(u2, v2, D)
            lo = max(-K2, lo1, lo2)
            hi = min(K2, hi1, hi2)
            if lo > hi:
                continue

            for k2 in range(lo, hi + 1):
                num_t = u1 - v1 * k2
                num_tp = u2 - v2 * k2
                if num_t == num_tp:
                    continue

                g = gcd_local(num_t, d1)
                add((l1 * (num_t // g)) * _SHIFT + (d1 // g))
                g = gcd_local(num_tp, d1)
                add((l1 * (num_tp // g)) * _SHIFT + (d1 // g))

    assert _key_for_integer_seconds(0) not in ambiguous
    assert _key_for_integer_seconds(1 * 3600 + 30 * 60) in ambiguous  # 1:30:00
    assert _key_for_integer_seconds(7 * 3600 + 30 * 60) in ambiguous  # 7:30:00
    assert _key_for_integer_seconds(3 * 3600) not in ambiguous  # 3:00:00
    assert _key_for_integer_seconds(9 * 3600) not in ambiguous  # 9:00:00

    print(len(ambiguous))


if __name__ == "__main__":
    main()
