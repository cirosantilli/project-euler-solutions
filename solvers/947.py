#!/usr/bin/env python3
"""Project Euler 947: Fibonacci Residues

Compute S(10^6) modulo 999,999,893.

No external libraries are used.
"""

import math


MOD = 999_999_893
N = 10**6


# -----------------------------
# Small brute-force checks
# -----------------------------


def _period_bruteforce(a: int, b: int, m: int) -> int:
    """Return p(a,b,m) for small m by simulating states (g(n), g(n+1))."""
    a %= m
    b %= m
    x, y = a, b
    # The state space has size m^2, so the period is <= m^2.
    for t in range(1, m * m + 2):
        x, y = y, (x + y) % m
        if x == a and y == b:
            return t
    raise RuntimeError("period not found")


def _s_bruteforce(m: int) -> int:
    total = 0
    for a in range(m):
        for b in range(m):
            p = _period_bruteforce(a, b, m)
            total += p * p
    return total


def _S_bruteforce(M: int) -> int:
    return sum(_s_bruteforce(m) for m in range(1, M + 1))


# Given in the problem statement
assert _s_bruteforce(3) == 513
assert _s_bruteforce(10) == 225_820
assert _S_bruteforce(3) == 542
assert _S_bruteforce(10) == 310_897


# -----------------------------
# Core number theory machinery
# -----------------------------


def _sieve_spf(limit: int):
    """Smallest prime factor sieve up to limit (inclusive)."""
    spf = list(range(limit + 1))
    spf[0] = 0
    if limit >= 1:
        spf[1] = 1
    r = int(limit**0.5)
    for i in range(2, r + 1):
        if spf[i] == i:  # prime
            step = i
            start = i * i
            for j in range(start, limit + 1, step):
                if spf[j] == j:
                    spf[j] = i
    return spf


def _fib_pair(n: int, mod: int):
    """Return (F_n, F_{n+1}) modulo mod using fast doubling (iterative)."""
    a, b = 0, 1
    # Process bits from MSB to LSB
    for i in range(n.bit_length() - 1, -1, -1):
        # Doubling formulas
        two_b_minus_a = (2 * b - a) % mod
        c = (a * two_b_minus_a) % mod  # F_{2k}
        d = (a * a + b * b) % mod  # F_{2k+1}
        if (n >> i) & 1:
            a, b = d, (c + d) % mod
        else:
            a, b = c, d
    return a, b


def _check_A_order(n: int, p: int) -> bool:
    """Check whether the Fibonacci Q-matrix A satisfies A^n = I (mod p)."""
    fn, fn1 = _fib_pair(n, p)
    return fn == 0 and fn1 == 1


def _pisano_prime(p: int, spf) -> int:
    """Pisano period modulo a prime p (order of Q-matrix A in GL(2, F_p))."""
    if p == 2:
        return 3
    if p == 5:
        return 20

    # Legendre symbol (5/p): pow returns 1 for residue, p-1 for non-residue.
    residue = pow(5, (p - 1) // 2, p) == 1
    candidate = (p - 1) if residue else (2 * (p + 1))

    # Factor candidate (distinct primes)
    x = candidate
    primes = []
    while x > 1:
        q = spf[x]
        primes.append(q)
        while x % q == 0:
            x //= q

    d = candidate
    for q in primes:
        while d % q == 0:
            nd = d // q
            if _check_A_order(nd, p):
                d = nd
            else:
                break
    return d


def _has_short_period_factor(p: int, pi_p: int) -> int:
    """Return k where some primitive pairs have period pi_p/k modulo p.

    For this problem, k is:
      - 5 when p=5
      - 2 for some primes where 5 is a quadratic residue
      - 1 otherwise

    The check uses det(A^n - I) = 1 + (-1)^n - L_n.
    """
    if p == 5:
        return 5
    if p == 2:
        return 1

    # Only possible when 5 is a quadratic residue (split case)
    if pow(5, (p - 1) // 2, p) != 1:
        return 1

    if pi_p % 2 != 0:
        return 1

    n = pi_p // 2
    fn, fn1 = _fib_pair(n, p)
    # Lucas L_n = F_{n-1}+F_{n+1} = 2*F_{n+1} - F_n
    ln = (2 * fn1 - fn) % p
    minus1_pow = 1 if (n % 2 == 0) else (p - 1)
    det = (1 + minus1_pow - ln) % p
    return 2 if det == 0 else 1


def _prime_power_distribution(p: int, e: int, pi_p: int, k: int):
    """Distribution of periods among primitive pairs modulo p^e.

    Returns a list of (period, count_mod_MOD).
    """
    # Pisano period for p^e
    pe1 = pow(p, e - 1)
    T = pi_p * pe1

    # Total number of primitive pairs mod p^e: p^{2e} - p^{2(e-1)}
    p2 = p * p
    total = pow(p2, e - 1) * (p2 - 1)

    if k == 1:
        return [(T, total % MOD)]

    if k == 2:
        small_period = T // 2
        small_count = (p - 1) * pe1
    else:
        # k == 5 (only for p=5)
        small_period = T // 5
        small_count = (p - 1) * pow(p2, e - 1)

    big_count = total - small_count
    return [
        (small_period, small_count % MOD),
        (T, big_count % MOD),
    ]


def solve() -> int:
    max_spf = 2 * N + 2
    spf = _sieve_spf(max_spf)

    # Precompute pi(p) and the short-period factor k for primes up to N.
    pi_prime = [0] * (N + 1)
    k_prime = [1] * (N + 1)

    for p in range(2, N + 1):
        if spf[p] == p:
            pi_p = _pisano_prime(p, spf)
            pi_prime[p] = pi_p
            k_prime[p] = _has_short_period_factor(p, pi_p)

    dist_cache = {}
    gcd = math.gcd

    ans = 0
    for n in range(1, N + 1):
        x = n
        dists = []
        while x > 1:
            p = spf[x]
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            key = (p, e)
            dist = dist_cache.get(key)
            if dist is None:
                dist = _prime_power_distribution(p, e, pi_prime[p], k_prime[p])
                dist_cache[key] = dist
            dists.append(dist)

        # Combine prime-power distributions via CRT (lcm of periods, product of counts).
        cur = {1: 1}
        for dist in dists:
            new = {}
            for per1, c1 in cur.items():
                for per2, c2 in dist:
                    g = gcd(per1, per2)
                    l = (per1 // g) * per2
                    v = (c1 * c2) % MOD
                    new[l] = (new.get(l, 0) + v) % MOD
            cur = new

        Pn = 0
        for per, c in cur.items():
            Pn = (Pn + (per * per) % MOD * c) % MOD

        ans = (ans + Pn * (N // n)) % MOD

    return ans


if __name__ == "__main__":
    print(solve())
