#!/usr/bin/env python3
"""
Project Euler 536

Find S(10^12), where S(n) is the sum of all m <= n such that
    a^(m+4) ≡ a (mod m)
for every integer a.

Constraints: single core, no external libraries.
"""

from math import gcd, isqrt
import sys


def sieve(limit: int):
    """Return (is_prime, primes) for 0..limit using a fast odd-only sieve."""
    if limit < 2:
        return bytearray(limit + 1), []
    is_prime = bytearray(b"\x01") * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    if limit >= 4:
        is_prime[4::2] = b"\x00" * (((limit - 4) // 2) + 1)  # clear even > 2

    r = isqrt(limit)
    for p in range(3, r + 1, 2):
        if is_prime[p]:
            step = p << 1
            start = p * p
            is_prime[start : limit + 1 : step] = b"\x00" * (
                ((limit - start) // step) + 1
            )

    primes = [2]
    primes.extend([p for p in range(3, limit + 1, 2) if is_prime[p]])
    return is_prime, primes


def S(N: int) -> int:
    """
    Sum of all m <= N satisfying a^(m+4) ≡ a (mod m) for all integers a.
    """
    if N < 1:
        return 0

    # Any prime divisor p of any valid m <= N satisfies p <= 2 + sqrt(N+4).
    pmax = isqrt(N + 4) + 5
    is_prime, primes = sieve(pmax)

    # Work only with odd primes. Handle m=2 separately (the only even solution).
    odd_primes = [p for p in primes if p >= 3]
    primes_after3 = [p for p in odd_primes if p != 3]

    # If 3 ∤ m then 3 ∤ (m+3), so no prime factor p can have 3 | (p-1),
    # hence every prime factor must be ≡ 2 (mod 3).
    primes_mod3_2 = [p for p in primes_after3 if (p % 3) == 2]

    total = 0
    if N >= 2:
        total += 2  # only even solution

    g = gcd
    sys.setrecursionlimit(20000)

    def feasible_progression(x: int, lam: int, min_q: int):
        """
        Solve x*q ≡ -3 (mod lam) for q, returning (step, r0, q0) where:
          q ≡ r0 (mod step), step = lam / gcd(x, lam),
          q0 is the smallest q >= min_q satisfying the congruence.
        Returns None if unsatisfiable.
        """
        gx = g(x, lam)
        if 3 % gx:
            return None
        step = lam // gx
        if step == 1:
            r0 = 0
        else:
            a = x // gx
            try:
                inv = pow(a, -1, step)
            except ValueError:
                return None
            r0 = ((-3 // gx) % step) * inv % step

        q0 = r0
        if q0 < min_q:
            q0 += ((min_q - q0 + step - 1) // step) * step
        return step, r0, q0

    def leaf_count_last_prime(
        x: int, lam: int, plist, idx: int, q_low: int, step: int, r0: int
    ):
        """
        Count solutions of the form m = x*q where q is a single final prime factor.
        This function assumes:
          - q must be >= plist[idx] (squarefree + increasing order)
          - q must be >= q_low (we only count primes too large to allow another factor)
          - q*x <= N
          - existing-prime constraints are enforced by q ≡ r0 (mod step)
        Additionally, the new prime q must satisfy (q-1) | (x+3).
        """
        nonlocal total

        if idx >= len(plist):
            return

        if q_low < plist[idx]:
            q_low = plist[idx]

        max_q = N // x
        if max_q > pmax:
            max_q = pmax
        if q_low > max_q:
            return

        # step==1 only happens for lam==1 (empty prime set), so congruence is trivial.
        if step == 1:
            # Iterate primes directly (there are few in the range because q_low > sqrt(N/x)).
            # Use manual lower-bound search.
            lo, hi = idx, len(plist)
            while lo < hi:
                mid = (lo + hi) >> 1
                if plist[mid] < q_low:
                    lo = mid + 1
                else:
                    hi = mid
            for j in range(lo, len(plist)):
                q = plist[j]
                if q > max_q:
                    break
                if (x + 3) % (q - 1) == 0:
                    total += x * q
            return

        # For very small step, scanning all integers is too dense; iterate primes and filter by residue.
        if step <= 2:
            lo, hi = idx, len(plist)
            while lo < hi:
                mid = (lo + hi) >> 1
                if plist[mid] < q_low:
                    lo = mid + 1
                else:
                    hi = mid
            for j in range(lo, len(plist)):
                q = plist[j]
                if q > max_q:
                    break
                if (q - r0) % step:
                    continue
                if (x + 3) % (q - 1) == 0:
                    total += x * q
            return

        # General case: scan the arithmetic progression q = r0 + k*step.
        q = r0
        if q < q_low:
            q += ((q_low - q + step - 1) // step) * step
        for cand in range(q, max_q + 1, step):
            if is_prime[cand] and (x + 3) % (cand - 1) == 0:
                total += x * cand

    def dfs(x: int, lam: int, plist, idx: int):
        nonlocal total

        # If lam | (x+3), then x already satisfies the criterion for all its prime factors.
        if (x + 3) % lam == 0:
            total += x

        if idx >= len(plist):
            return

        # If we add a new prime q > sqrt(N/x), then q must be the last prime factor
        # (because x*q*q > N). We count these in a dedicated routine.
        bound = isqrt(N // x)
        q_low = bound + 1

        # Feasibility check for any extension: remaining multiplier must satisfy x*y ≡ -3 (mod lam).
        prog = feasible_progression(x, lam, plist[idx])
        if prog is None:
            return
        step, r0, q0 = prog
        if x * q0 > N:
            return

        if q_low <= pmax:
            leaf_count_last_prime(x, lam, plist, idx, q_low, step, r0)

        # Recurse by picking the next prime factor p <= bound (so there remains room for at least one more prime).
        for j in range(idx, len(plist)):
            p = plist[j]
            if p > bound:
                break

            xp = x * p
            pm1 = p - 1

            d = g(lam, pm1)
            lam2 = (lam // d) * pm1  # lcm(lam, p-1)

            # Any extendable partial product must satisfy gcd(xp, lam2) in {1,3}.
            gp = g(xp, lam2)
            if gp != 1 and gp != 3:
                continue

            dfs(xp, lam2, plist, j + 1)

    # Branch A: m divisible by 3 (start with factor 3).
    if N >= 3:
        # For x=3, lam=lcm(3-1)=2, and we may add primes >= 5.
        dfs(3, 2, primes_after3, 0)

    # Branch B: m not divisible by 3 => all primes are ≡ 2 (mod 3).
    dfs(1, 1, primes_mod3_2, 0)

    return total


def main():
    # Test values from the problem statement:
    assert S(100) == 32
    assert S(10**6) == 22868117

    print(S(10**12))


if __name__ == "__main__":
    main()
