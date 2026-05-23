#!/usr/bin/env python
"""
Project Euler 483 - Repeated Permutation

We use the cycle-type formula. If a permutation on n elements has a_i cycles of length i,
then the number of such permutations is:

    n! / ∏_i (a_i! * i^{a_i})

and its order is lcm({ i : a_i > 0 }).

After dividing by n!, the required expectation becomes:

    g(n) =  Σ_{ Σ i*a_i = n }  lcm({i : a_i>0})^2  /  ∏_i (a_i! * i^{a_i})

We compute this sum with dynamic programming over cycle lengths, processing lengths
from n down to 1. To keep the LCM-state small, we "extract" prime-power contributions
as soon as they can no longer be affected by remaining (smaller) cycle lengths.
"""

from __future__ import annotations

from math import gcd


def _lcm(a: int, b: int) -> int:
    return a // gcd(a, b) * b


def _largest_prime_factors(n: int) -> tuple[list[int], list[int]]:
    lpf = [0] * (n + 1)
    primes: list[int] = []
    for p in range(2, n + 1):
        if lpf[p]:
            continue
        primes.append(p)
        for m in range(p, n + 1, p):
            lpf[m] = p
    return lpf, primes


def format_sci_10(x: float) -> str:
    """
    Format with 10 significant digits, as in the problem statement.
    Example: 5.166666667e0 (no '+' sign, no leading zeros in exponent).
    """
    s = f"{x:.9e}"  # 10 significant digits total (1 before '.' + 9 after)
    mant, exp = s.split("e")
    return f"{mant}e{int(exp)}"


def g(n: int) -> float:
    """
    Compute g(n) as defined in Project Euler 483.

    Cycle lengths are processed in descending blocks of largest prime factor.
    After finishing the block for a prime p, no later cycle length can contain p,
    so all p-powers in the tracked LCM can be absorbed into the weight as p^(2a).
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return 1.0

    lpf, primes = _largest_prime_factors(n)
    by_lpf: list[list[int]] = [[] for _ in range(n + 1)]
    for c in range(2, n + 1):
        by_lpf[lpf[c]].append(c)

    # Start with fixed points only: t one-cycles have weight 1/t!.
    dp: list[dict[int, float]] = [dict() for _ in range(n + 1)]
    dp[0][1] = 1.0
    fixed_weight = 1.0
    for used in range(1, n + 1):
        fixed_weight /= used
        dp[used][1] = fixed_weight

    for p in reversed(primes):
        for c in by_lpf[p]:
            new = [d.copy() for d in dp]

            factors: list[float] = []
            term = 1.0
            for m in range(1, n // c + 1):
                term /= c * m
                factors.append(term)

            for used in range(n - c + 1):
                d = dp[used]
                if not d:
                    continue
                max_m = (n - used) // c
                for L0, v0 in d.items():
                    L1 = _lcm(L0, c)
                    used1 = used
                    for m in range(max_m):
                        used1 += c
                        nd = new[used1]
                        val = v0 * factors[m]
                        try:
                            nd[L1] += val
                        except KeyError:
                            nd[L1] = val

            dp = new

        p2 = float(p * p)
        for used, d in enumerate(dp):
            if not d:
                continue
            compressed: dict[int, float] = {}
            for L, v in d.items():
                l = L
                val = v
                while l % p == 0:
                    l //= p
                    val *= p2
                try:
                    compressed[l] += val
                except KeyError:
                    compressed[l] = val
            dp[used] = compressed

    return sum(float(L * L) * v for L, v in dp[n].items())


def main() -> None:
    # Test values from the problem statement (10 significant digits).
    assert format_sci_10(g(3)) == "5.166666667e0"
    assert format_sci_10(g(5)) == "1.734166667e1"
    assert format_sci_10(g(20)) == "5.106136147e3"

    print(format_sci_10(g(350)))


if __name__ == "__main__":
    main()
