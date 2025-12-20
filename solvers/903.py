#!/usr/bin/env python3
"""
Project Euler 903: Total Permutation Powers

Compute Q(n) modulo 1_000_000_007.

Requirements satisfied:
- No external libraries.
- Asserts for all sample/test values from the problem statement.
- Does NOT hardcode/mention the final required answer anywhere; it is only printed.
"""

from __future__ import annotations

import sys
from array import array

MOD = 1_000_000_007


def brute_q(n: int) -> int:
    """Brute force Q(n) for very small n (used only for n <= 7)."""
    import itertools

    def compose(p, q):
        # p ∘ q in one-line notation (apply q then p)
        return tuple(p[q[i] - 1] for i in range(n))

    def rank_perm(p):
        # factoradic rank (1-based)
        elems = list(range(1, n + 1))
        r = 1
        fact = 1
        # precompute factorial weights
        weights = [1] * (n + 1)
        for i in range(2, n + 1):
            weights[i] = weights[i - 1] * i
        for i, x in enumerate(p):
            idx = elems.index(x)
            r += idx * weights[n - i - 1]
            elems.pop(idx)
        return r

    perms = list(itertools.permutations(range(1, n + 1)))
    factn = 1
    for i in range(2, n + 1):
        factn *= i

    total = 0
    for pi in perms:
        cur = tuple(range(1, n + 1))  # pi^0
        for _ in range(factn):
            cur = compose(pi, cur)
            total += rank_perm(cur)
    return total


def mobius_sieve(n: int) -> array:
    """Compute Möbius mu[1..n] using a linear sieve."""
    mu = array("b", [0]) * (n + 1)
    mu[1] = 1
    primes = []
    is_comp = bytearray(n + 1)

    for i in range(2, n + 1):
        if not is_comp[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            v = i * p
            if v > n:
                break
            is_comp[v] = 1
            if i % p == 0:
                mu[v] = 0
                break
            mu[v] = -mu[i]
    return mu


def compute_q(n: int) -> int:
    """
    Compute Q(n) mod MOD.

    For n >= 4 the main O(n log n) algorithm is used.
    For very small n we brute-force to avoid corner-case divisions.
    """
    if n <= 7:
        return brute_q(n) % MOD

    # modular inverses inv[i] for i <= n
    inv = array("I", [0]) * (n + 1)
    inv[1] = 1
    for i in range(2, n + 1):
        inv[i] = MOD - (MOD // i) * inv[MOD % i] % MOD

    # harmonic numbers H[k] = sum_{i=1..k} inv[i]
    H = array("I", [0]) * (n + 1)
    s = 0
    for i in range(1, n + 1):
        s += inv[i]
        if s >= MOD:
            s -= MOD
        H[i] = s

    # Möbius function
    mu = mobius_sieve(n)

    # Compute F[s] = sum_{d|s} mu(d)/d * H[s/d - 1]
    # by convolution over multiples.
    F = array("I", [0]) * (n + 1)
    H_local = H
    inv_local = inv
    F_local = F

    for d in range(1, n + 1):
        md = mu[d]
        if md == 0:
            continue
        c = inv_local[d] if md == 1 else (MOD - inv_local[d])
        m = 1
        for s2 in range(d, n + 1, d):
            F_local[s2] = (F_local[s2] + c * H_local[m - 1]) % MOD
            m += 1

    # S = sum_{s=2..n} H[floor(n/s)] * 2*F[s]/s
    S = 0
    for s2 in range(2, n + 1):
        term = H_local[n // s2] * ((2 * F_local[s2]) % MOD) % MOD
        term = term * inv_local[s2] % MOD
        S += term
        S %= MOD

    # alpha = probability two given elements are both fixed in sigma = pi^k
    num_alpha = (n % MOD - H_local[n] + S) % MOD
    alpha = num_alpha * inv_local[n] % MOD * inv_local[n - 1] % MOD

    # beta = probability two given elements are swapped
    denom_beta = (2 * n * (n - 1)) % MOD
    beta = H_local[n // 2] * pow(denom_beta, MOD - 2, MOD) % MOD

    # p = P(sigma(x) = x) for a given x
    p = H_local[n] * inv_local[n] % MOD
    # q = P(sigma(x) = y) for a given x and y != x
    q = (1 - p) % MOD * inv_local[n - 1] % MOD

    inv_n2 = inv_local[n - 2]
    inv_n3 = inv_local[n - 3]

    # a = P(A=other, B=special) parameter from derivation
    a = (p - alpha) % MOD * inv_n2 % MOD
    # b = another parameter from derivation
    b = (q - beta) % MOD * inv_n2 % MOD
    # eta = bulk uniform parameter
    eta = (q - a - b) % MOD * inv_n3 % MOD

    # P_d is linear in d with slope (b-a), so rank expectation needs only
    # two factorial-weighted sums.
    bconst = ((n - 2) * (n - 3) // 2) % MOD
    C0 = (beta + (n - 3) % MOD * b + (n - 1) % MOD * a + eta * bconst) % MOD
    slope = (b - a) % MOD

    inv2 = (MOD + 1) // 2

    # Compute S1 = sum_{m=1..n-1} m! * m
    # and S2 = sum_{m=1..n-1} m! * m(m+1)/2
    fact = 1
    S1 = 0
    S2 = 0
    for m in range(1, n):
        fact = (fact * m) % MOD  # now fact = m!
        S1 = (S1 + fact * m) % MOD
        S2 = (S2 + fact * m % MOD * (m + 1) % MOD * inv2) % MOD

    fact_n = fact * n % MOD  # n!
    E_rank = (1 + C0 * S1 + slope * S2) % MOD

    return fact_n * fact_n % MOD * E_rank % MOD


def main() -> None:
    # Asserts for ALL sample values from the problem statement.
    assert compute_q(2) == 5
    assert compute_q(3) == 88
    assert compute_q(6) == 133103808
    assert compute_q(10) == 468421536

    n = 1_000_000
    if len(sys.argv) > 1:
        n = int(sys.argv[1])

    print(compute_q(n))


if __name__ == "__main__":
    main()
