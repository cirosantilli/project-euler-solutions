#!/usr/bin/env python3
"""Project Euler 920: Tau Numbers

We use the divisor-count function tau(n).
A positive integer n is a *tau number* if tau(n) divides n.

Define:
- m(k): the smallest tau number x such that tau(x) = k
- M(n): the sum of all m(k) whose values do not exceed 10^n

This script computes and prints M(16).

Constraints:
- No third-party libraries are used.
- Any known final answer is NOT embedded; it is computed and printed.
"""

from __future__ import annotations


LIMIT = 10**16


def sieve(limit: int) -> list[int]:
    """Return all primes <= limit via a basic sieve."""
    if limit < 2:
        return []
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    r = int(limit**0.5)
    for p in range(2, r + 1):
        if is_prime[p]:
            step = p
            start = p * p
            is_prime[start : limit + 1 : step] = [False] * (
                ((limit - start) // step) + 1
            )
    return [i for i, v in enumerate(is_prime) if v]


PRIMES = sieve(200)


def precompute_small_factorizations(max_n: int) -> list[dict[int, int]]:
    """Factorize every integer in [0..max_n] using trial division by small primes."""
    fac = [dict() for _ in range(max_n + 1)]
    for x in range(2, max_n + 1):
        n = x
        d: dict[int, int] = {}
        for p in PRIMES:
            if p * p > n:
                break
            if n % p == 0:
                e = 0
                while n % p == 0:
                    n //= p
                    e += 1
                d[p] = e
        if n > 1:
            d[n] = d.get(n, 0) + 1
        fac[x] = d
    return fac


def generate_exponent_vectors(limit: int) -> list[tuple[int, ...]]:
    """Generate all non-increasing exponent vectors a1>=a2>=...>=ar>=1 such that

        (2^a1)*(3^a2)*...*(p_r^ar) <= limit,

    where p_i is the i-th prime.

    This enumerates all exponent *multisets* that could correspond to a minimal candidate,
    since swapping larger primes downward only decreases the number.
    """
    # Determine how many primes are needed for the lower-bound construction.
    primes_lb: list[int] = []
    prod = 1
    for p in PRIMES:
        if prod * p > limit:
            break
        prod *= p
        primes_lb.append(p)

    out: list[tuple[int, ...]] = []
    exps: list[int] = []

    # Using recursion depth <= len(primes_lb) (<= 13 for 1e16).
    def dfs(idx: int, max_e: int, current: int) -> None:
        if idx >= len(primes_lb):
            return
        p = primes_lb[idx]
        p_pow = 1
        for e in range(1, max_e + 1):
            p_pow *= p
            nxt = current * p_pow
            if nxt > limit:
                break
            exps.append(e)
            out.append(tuple(exps))
            dfs(idx + 1, e, nxt)
            exps.pop()

    dfs(0, 60, 1)
    return out


def min_tau_number_for_exponents(
    exps_desc: tuple[int, ...],
    req: dict[int, int],
    primes_sorted: list[int],
    limit: int,
) -> int | None:
    """For a fixed exponent multiset (sorted non-increasing), minimize

        n = Π p_i^{a_i}

    subject to:
    - tau(n) is fixed by the exponents, so tau(n) = k
    - k divides n

    The constraint k|n means: for each prime q dividing k with exponent t=v_q(k),
    the prime q must be among the chosen primes for n, and the exponent on q in n
    must be at least t.

    We search assignments of exponents to the *required primes* (prime factors of k).
    The remaining exponents are greedily assigned (largest-to-smallest) to the
    remaining (smallest possible) filler primes.
    """
    r = len(exps_desc)
    if r == 0:
        return 1

    required_primes = [p for p in req.keys()]
    s = len(required_primes)
    if s > r:
        return None

    max_exp = exps_desc[0]
    for p, need_exp in req.items():
        if need_exp > max_exp:
            return None

    required_set = set(required_primes)

    # Smallest filler primes not in required_set.
    fillers_needed = r - s
    fillers: list[int] = []
    if fillers_needed:
        for p in primes_sorted:
            if p not in required_set:
                fillers.append(p)
                if len(fillers) == fillers_needed:
                    break

    fillers.sort()

    # Decide which exponent goes to each required prime.
    # Process larger required primes earlier (they prefer smaller exponents).
    required_primes.sort(reverse=True)

    exps = list(exps_desc)
    all_mask = (1 << r) - 1

    # Greedy upper bound to help pruning.
    def greedy_upper_bound() -> int | None:
        avail = sorted(exps)  # ascending
        prod = 1
        for p in required_primes:
            need = req[p]
            for i, e in enumerate(avail):
                if e >= need:
                    prod *= pow(p, e)
                    avail.pop(i)
                    break
            else:
                return None
            if prod > limit:
                return None
        avail.sort(reverse=True)
        for p, e in zip(fillers, avail):
            prod *= pow(p, e)
            if prod > limit:
                return None
        return prod

    best = greedy_upper_bound()
    if best is None:
        best_val = None
        best_int = 10**100
    else:
        best_val = best
        best_int = best

    # Lower bound ignoring requirements for remaining required primes.
    def lower_bound(mask: int, current_prod: int, next_required_idx: int) -> int:
        rem_exps = [exps[i] for i in range(r) if not ((mask >> i) & 1)]
        rem_exps.sort(reverse=True)
        rem_primes = fillers + required_primes[next_required_idx:]
        rem_primes.sort()
        lb = current_prod
        for p, e in zip(rem_primes, rem_exps):
            lb *= pow(p, e)
            if lb >= best_int:
                break
        return lb

    def dfs(req_idx: int, mask: int, current_prod: int) -> None:
        nonlocal best_int, best_val

        if current_prod >= best_int:
            return
        if req_idx == len(required_primes):
            # Assign remaining exponents to fillers.
            rem_exps = [exps[i] for i in range(r) if not ((mask >> i) & 1)]
            rem_exps.sort(reverse=True)
            total = current_prod
            for p, e in zip(fillers, rem_exps):
                total *= pow(p, e)
                if total >= best_int:
                    return
            if total <= limit and total < best_int:
                best_int = total
                best_val = total
            return

        # Prune with a cheap lower bound.
        if lower_bound(mask, current_prod, req_idx) >= best_int:
            return

        p = required_primes[req_idx]
        need = req[p]

        prev_e = None
        for i in range(r):
            if (mask >> i) & 1:
                continue
            e = exps[i]
            if e < need:
                continue
            if prev_e == e:
                continue
            prev_e = e

            nxt = current_prod * pow(p, e)
            if nxt >= best_int or nxt > limit:
                continue
            dfs(req_idx + 1, mask | (1 << i), nxt)

    dfs(0, 0, 1)
    return best_val


def compute_m_values(limit: int) -> dict[int, int]:
    """Compute m(k) for all k where m(k) <= limit."""
    exponent_vectors = generate_exponent_vectors(limit)

    # Maximum possible (a+1) encountered determines factor table size.
    max_a = 1
    for v in exponent_vectors:
        if v and v[0] > max_a:
            max_a = v[0]
    small_fac = precompute_small_factorizations(max_a + 2)

    best: dict[int, int] = {1: 1}

    for exps in exponent_vectors:
        # tau = product(a_i+1) and its prime factorization.
        k = 1
        req: dict[int, int] = {}
        for a in exps:
            k *= a + 1
            for p, e in small_fac[a + 1].items():
                req[p] = req.get(p, 0) + e

        n = min_tau_number_for_exponents(exps, req, PRIMES, limit)
        if n is None:
            continue
        if n <= limit:
            prev = best.get(k)
            if prev is None or n < prev:
                best[k] = n

    return best


def main() -> None:
    best = compute_m_values(LIMIT)

    # Tests from the problem statement.
    assert best[8] == 24
    assert best[12] == 60
    assert best[16] == 384

    m3 = sum(v for v in best.values() if v <= 10**3)
    assert m3 == 3189

    ans = sum(best.values())
    print(ans)


if __name__ == "__main__":
    main()
