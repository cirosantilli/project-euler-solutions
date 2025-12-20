#!/usr/bin/env python3
"""
Project Euler 864 - Square + 1 = Squarefree

Counts squarefree values among x^2 + 1 for 1 <= x <= n.
No third-party libraries are used.

Performance note:
The tail-correction uses negative Pell equations x^2 - k*y^2 = -1.
We only care about solutions with x <= n, so the continued-fraction solver
is bounded: it exits early as soon as convergents exceed n.
"""

import sys
import math
from array import array


# ----------------------------
# Small brute helpers (tests)
# ----------------------------


def is_squarefree(m: int) -> bool:
    """Return True iff m is squarefree (trial division; for small inputs)."""
    p = 2
    while p * p <= m:
        if m % p == 0:
            m //= p
            if m % p == 0:
                return False
        p += 1 if p == 2 else 2
    return True


def brute_C(n: int) -> int:
    """Brute force C(n) (only for tiny n; used for statement asserts)."""
    c = 0
    for x in range(1, n + 1):
        if is_squarefree(x * x + 1):
            c += 1
    return c


# ----------------------------
# Prime sieve (odd-only)
# ----------------------------


def primes_upto(limit: int):
    """Return list of all primes <= limit."""
    if limit < 2:
        return []
    if limit == 2:
        return [2]
    size = limit // 2 + 1  # index i represents 2*i+1
    sieve = bytearray(b"\x01") * size
    sieve[0] = 0  # 1 is not prime
    r = int(math.isqrt(limit))
    for p in range(3, r + 1, 2):
        if sieve[p // 2]:
            start = (p * p) // 2
            step = p
            sieve[start::step] = b"\x00" * (((size - 1 - start) // step) + 1)
    out = [2]
    for i in range(1, size):
        if sieve[i]:
            out.append(2 * i + 1)
    return out


def primes_1mod4_upto(limit: int):
    """Return primes p <= limit with p % 4 == 1 as array('I')."""
    ps = primes_upto(limit)
    out = array("I")
    for p in ps:
        if p % 4 == 1:
            out.append(p)
    return out


# ----------------------------
# Roots of x^2 == -1 (mod p^2)
# ----------------------------


def sqrt_minus_one_mod_p(p: int) -> int:
    """Return r such that r^2 == -1 (mod p), for odd prime p == 1 (mod 4).

    Pick a quadratic non-residue g mod p, then:
      g^((p-1)/2) == -1 (mod p)
    hence:
      r = g^((p-1)/4) (mod p) satisfies r^2 == -1 (mod p).
    """
    exp = (p - 1) // 2
    g = 2
    while pow(g, exp, p) != p - 1:
        g += 1
    return pow(g, (p - 1) // 4, p)


def roots_minus_one_mod_p2(p: int):
    """For odd prime p == 1 (mod 4), return the two solutions to x^2 == -1 (mod p^2)."""
    r = sqrt_minus_one_mod_p(p)
    # Hensel lift r from mod p to mod p^2 for f(x)=x^2+1.
    p2 = p * p
    s = (r * r + 1) // p  # exact since r^2 == -1 (mod p) and 0 <= r < p
    inv = pow((2 * r) % p, -1, p)
    t = (-s * inv) % p
    R = (r + t * p) % p2
    return (R, (-R) % p2)


def crt_combine(residues, m: int, a1: int, a2: int, p2: int):
    """Combine x == r (mod m) with x == a (mod p2) for a in {a1,a2}.

    Assumes gcd(m, p2) = 1.
    Returns the combined residues modulo m*p2.
    """
    inv = pow(m % p2, -1, p2)  # inverse of m modulo p2
    out = []
    for r in residues:
        t = ((a1 - r) * inv) % p2
        out.append(r + m * t)
        t = ((a2 - r) * inv) % p2
        out.append(r + m * t)
    return out


# ----------------------------
# Direct Mobius sum up to D
# ----------------------------


def direct_mobius_sum(n: int, D: int) -> int:
    """Compute S_D(n) = sum_{d<=D} mu(d) * A_d(n) for admissible d.

    Admissible means: d is squarefree and every odd prime factor p of d satisfies p == 1 (mod 4).
    (If d has any odd p == 3 (mod 4), then x^2 == -1 (mod p) has no solutions and A_d(n)=0.
     If 2|d then 4|d^2, but 4 never divides x^2+1, so A_d(n)=0 as well.)

    We enumerate admissible d by DFS over primes p == 1 (mod 4), maintaining the solution set
    of x^2 == -1 (mod d^2) as residues modulo d^2.
    """
    primes = primes_1mod4_upto(D)

    # Precompute two roots mod p^2 for each prime
    roots = array("Q")  # store [r1,r2,r1,r2,...]
    for p in primes:
        r1, r2 = roots_minus_one_mod_p2(int(p))
        roots.append(r1)
        roots.append(r2)

    ans = n  # d=1 term (mu(1)=1, A_1(n)=n)
    P = len(primes)
    sys.setrecursionlimit(1_000_000)

    def dfs(start_idx: int, d: int, mod: int, residues, mu_sign: int):
        nonlocal ans
        for i in range(start_idx, P):
            p = int(primes[i])
            nd = d * p
            if nd > D:
                break

            p2 = p * p
            a1 = int(roots[2 * i])
            a2 = int(roots[2 * i + 1])
            nmod = mod * p2
            nres = crt_combine(residues, mod, a1, a2, p2)
            nmu = -mu_sign

            # Count A_nd(n) = #{x in [1..n] : x == r (mod nmod) for some r in nres}
            # For admissible d>1, residues never include 0, so counting [0..n]
            # is identical to counting [1..n].
            if nmod <= n:
                q, rem = divmod(n, nmod)
                A = q * len(nres)
                cnt = 0
                for r in nres:
                    if r <= rem:
                        cnt += 1
                A += cnt
            else:
                A = 0
                for r in nres:
                    if r <= n:
                        A += 1

            ans += nmu * A
            dfs(i + 1, nd, nmod, nres, nmu)

    dfs(0, 1, 1, [0], 1)
    return ans


# ----------------------------
# Negative Pell tail correction (bounded CF)
# ----------------------------


def negative_pell_fundamental_bounded(D: int, x_limit: int):
    """Return minimal (x,y) solving x^2 - D*y^2 = -1, or None if no useful solution.

    Uses the continued fraction of sqrt(D). The negative Pell has a solution iff
    the CF period length is odd.

    Optimization: we only need solutions with x <= x_limit. Convergent numerators
    p_i grow strictly, so once p_i > x_limit, later convergents cannot help and
    we can stop early.
    """
    a0 = int(math.isqrt(D))
    if a0 * a0 == D:
        return None

    m = 0
    d = 1
    a = a0

    # convergents: p_{-1}=1, p_0=a0; q_{-1}=0, q_0=1
    p_prev, p = 1, a0
    q_prev, q = 0, 1

    period = 0
    while True:
        m = d * a - m
        d = (D - m * m) // d
        a = (a0 + m) // d

        p_prev, p = p, a * p + p_prev
        q_prev, q = q, a * q + q_prev

        period += 1
        if p_prev > x_limit:
            return None
        if a == 2 * a0:
            break

    if period % 2 == 0:
        return None

    x, y = p_prev, q_prev
    if x * x - D * y * y != -1:
        return None
    return (x, y)


def primes_for_factoring(limit: int):
    """Convenience: primes up to limit (used to factor y values)."""
    return primes_upto(limit)


def factor_distinct_primes(n: int, primes_small):
    """Return distinct prime factors of n using trial division by primes_small."""
    out = []
    t = n
    for p in primes_small:
        if p * p > t:
            break
        if t % p == 0:
            out.append(p)
            while t % p == 0:
                t //= p
    if t > 1:
        out.append(t)
    return out


def mobius_tail_sum_for_y(y: int, D: int, primes_small) -> int:
    """Compute sum_{d|y, d>D} mu(d).

    Only squarefree divisors matter; we enumerate subset products of the distinct primes of y.
    """
    pf = factor_distinct_primes(y, primes_small)
    total = 0

    def rec(i: int, prod: int, parity: int):
        nonlocal total
        if i == len(pf):
            if prod > D:
                total += -1 if parity else 1
            return
        rec(i + 1, prod, parity)
        rec(i + 1, prod * pf[i], parity ^ 1)

    rec(0, 1, 0)
    return total


def correction_via_pell(n: int, D: int) -> int:
    """Exact correction for the truncation at d<=D."""
    Kmax = (n * n + 1) // (D * D)
    if Kmax < 2:
        return 0

    # Linear sieve up to Kmax: spf and mu
    spf = array("I", [0]) * (Kmax + 1)
    mu = array("b", [0]) * (Kmax + 1)
    primes = array("I")
    mu[1] = 1
    for i in range(2, Kmax + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
            mu[i] = -1
        for p in primes:
            ip = i * p
            if ip > Kmax:
                break
            spf[ip] = p
            if i % p == 0:
                mu[ip] = 0
                break
            mu[ip] = -mu[i]

    primes_small = primes_for_factoring(int(math.isqrt(n)) + 1)

    corr = 0
    spf_local = spf
    mu_local = mu

    for k in range(2, Kmax + 1):
        if mu_local[k] == 0:
            continue

        # Filter: if an odd prime p == 3 (mod 4) divides k, no solutions exist.
        t = k
        ok = True
        while t > 1:
            p = spf_local[t]
            t //= p
            if p != 2 and (p & 3) == 3:
                ok = False
                break
        if not ok:
            continue

        sol = negative_pell_fundamental_bounded(k, n)
        if sol is None:
            continue

        x, y = sol

        # Fundamental +1 unit is (x+y*sqrt(k))^2 = A + B*sqrt(k)
        A = x * x + k * y * y
        B = 2 * x * y

        while x <= n:
            if y > D:
                corr += mobius_tail_sum_for_y(y, D, primes_small)
            x, y = (A * x + k * B * y), (B * x + A * y)

    return corr


# ----------------------------
# Final assembly
# ----------------------------


def compute_C(n: int, D: int) -> int:
    """Compute C(n) exactly."""
    return direct_mobius_sum(n, D) + correction_via_pell(n, D)


def main() -> None:
    # Test values given in the problem statement
    assert brute_C(10) == 9
    assert brute_C(1000) == 895

    N = 123567101113

    # Tunable truncation bound.
    D = 30_000_000
    if len(sys.argv) >= 2:
        D = int(sys.argv[1])

    print(compute_C(N, D))


if __name__ == "__main__":
    main()
