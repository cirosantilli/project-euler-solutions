#!/usr/bin/env python
from math import gcd, log10
import sys

try:
    # Python 3.11+ protects int-to-string conversion by default.  The final
    # product has far more than 4300 digits, so allow this trusted conversion.
    sys.set_int_max_str_digits(0)
except AttributeError:
    pass

LIMIT = 20_000
PRIME_SEARCH_LIMIT = 2_000_000


def sieve(n):
    """Return all primes up to n, inclusive."""
    is_prime = bytearray(b"\x01") * (n + 1)
    if n >= 0:
        is_prime[0] = 0
    if n >= 1:
        is_prime[1] = 0

    r = int(n**0.5)
    for i in range(2, r + 1):
        if is_prime[i]:
            start = i * i
            is_prime[start : n + 1 : i] = b"\x00" * (((n - start) // i) + 1)
    return [i for i in range(n + 1) if is_prime[i]]


PRIMES = sieve(PRIME_SEARCH_LIMIT)
TARGET_PRIMES = [p for p in PRIMES if p < LIMIT]


def factor(n):
    """Prime factorization of n as (prime, exponent) pairs."""
    out = []
    t = n
    for p in PRIMES:
        if p * p > t:
            break
        if t % p == 0:
            e = 0
            while t % p == 0:
                t //= p
                e += 1
            out.append((p, e))
    if t > 1:
        out.append((t, 1))
    return out


def divisors_from_factorization(factors):
    """Sorted positive divisors from a prime factorization."""
    divs = [1]
    for p, e in factors:
        old = divs
        divs = []
        power = 1
        for _ in range(e + 1):
            for d in old:
                divs.append(d * power)
            power *= p
    return sorted(divs)


def primitive_root(p, prime_factors_of_p_minus_1):
    """Smallest primitive root modulo the prime p."""
    if p == 2:
        return 1
    m = p - 1
    for g in range(2, p):
        ok = True
        for q in prime_factors_of_p_minus_1:
            if pow(g, m // q, p) == 1:
                ok = False
                break
        if ok:
            return g
    raise RuntimeError("primitive root not found")


def discrete_log_table(p, root):
    """table[a] = k where root**k == a (mod p), for nonzero a."""
    table = [-1] * p
    x = 1
    for k in range(p - 1):
        table[x] = k
        x = (x * root) % p
    return table


S_CACHE = {}


def S_for_prime(p):
    """Return (S(p), log10(S(p))).  S(p) itself is computed exactly."""
    if p in S_CACHE:
        return S_CACHE[p]
    if p == 2:
        S_CACHE[p] = (1, 0.0)
        return S_CACHE[p]

    m = p - 1
    factors = factor(m)
    divs = divisors_from_factorization(factors)
    root = primitive_root(p, [q for q, _ in factors])
    dlog = discrete_log_table(p, root)

    # For a prime q, only c = gcd(log_root(q), m) matters for every quotient
    # order used by the dynamic program.  Record the least rational prime q for
    # each proper divisor c of m.
    needed_c_count = len(divs) - 1
    least_prime_for_c = {}
    for q in PRIMES:
        if q == p:
            continue
        c = gcd(dlog[q % p], m)
        if c < m and c not in least_prime_for_c:
            least_prime_for_c[c] = q
            if len(least_prime_for_c) == needed_c_count:
                break
    if len(least_prime_for_c) != needed_c_count:
        raise RuntimeError("increase PRIME_SEARCH_LIMIT")

    # Current covered subgroup has size h.  Let M = m / h.  Expanding by a
    # factor L requires the least prime q whose coset has exact order L in the
    # quotient of size M.  If d = M / L, this is gcd(log_root(q), M) = d.
    best_by_M = {}
    c_items = list(least_prime_for_c.items())
    for M in divs:
        if M == 1:
            continue
        best = {}
        for c, q in c_items:
            d = gcd(c, M)
            if d < M and (d not in best or q < best[d]):
                best[d] = q
        best_by_M[M] = best

    # Exact dynamic program over subgroup sizes h | m.  The value stored at h is
    # the smallest integer producing a direct divisor-residue tiling of that
    # subgroup; logs are carried only for the final scientific notation helper.
    dp_value = {1: 1}
    dp_log = {1: 0.0}
    for h in divs:
        if h not in dp_value:
            continue
        M = m // h
        if M == 1:
            continue
        best = best_by_M[M]
        base_value = dp_value[h]
        base_log = dp_log[h]
        for L in divs:
            if L > 1 and M % L == 0:
                next_h = h * L
                q = best[M // L]
                candidate = base_value * pow(q, L - 1)
                if next_h not in dp_value or candidate < dp_value[next_h]:
                    dp_value[next_h] = candidate
                    dp_log[next_h] = base_log + (L - 1) * log10(q)

    S_CACHE[p] = (dp_value[m], dp_log[m])
    return S_CACHE[p]


def product_T(limit):
    """Exact product of S(p) over primes p < limit."""
    product = 1
    for p in PRIMES:
        if p >= limit:
            break
        product *= S_for_prime(p)[0]
    return product


def scientific_from_int(n, places=5):
    """Scientific notation for a positive integer, rounded by decimal digits."""
    if n <= 0:
        raise ValueError("n must be positive")

    digits = str(n)
    exponent = len(digits) - 1
    significant = places + 1

    if len(digits) > significant:
        head = int(digits[:significant])
        if int(digits[significant]) >= 5:
            head += 1
        if head == 10**significant:
            head //= 10
            exponent += 1
    else:
        head = int(digits) * (10 ** (significant - len(digits)))

    mantissa_digits = f"{head:0{significant}d}"
    mantissa = mantissa_digits[0] + "." + mantissa_digits[1:]
    return f"{mantissa}e{exponent}"


def run_tests():
    assert S_for_prime(2)[0] == 1
    assert S_for_prime(5)[0] == 8
    assert product_T(20) == 1_348_422_598_656
    assert scientific_from_int(product_T(100)) == "1.37451e123"


def main():
    run_tests()
    print(scientific_from_int(product_T(LIMIT)))


if __name__ == "__main__":
    main()
