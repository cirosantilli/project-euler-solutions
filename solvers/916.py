#!/usr/bin/env python3
"""Project Euler 916: Restricted Permutations

Let P(n) be the number of permutations of {1,2,...,2n} such that:
  1) there is no increasing subsequence with more than n+1 elements, and
  2) there is no decreasing subsequence with more than 2 elements.

This program computes P(10^8) modulo 1_000_000_007.

No external libraries are used.
"""

from __future__ import annotations

import os
import sys

MOD = 1_000_000_007


def _split_inclusive(lo: int, hi: int, parts: int) -> list[tuple[int, int]]:
    """Split [lo, hi] into `parts` non-empty (or as-even-as-possible) subranges."""
    if lo > hi:
        return []
    length = hi - lo + 1
    parts = max(1, min(parts, length))
    base, rem = divmod(length, parts)
    out: list[tuple[int, int]] = []
    cur = lo
    for i in range(parts):
        seg_len = base + (1 if i < rem else 0)
        nxt = cur + seg_len - 1
        out.append((cur, nxt))
        cur = nxt + 1
    return out


def _prod_range_mod(lo: int, hi: int, mod: int) -> int:
    """Compute (lo * (lo+1) * ... * hi) % mod for lo<=hi."""
    acc = 1
    x = lo
    # while-loop is slightly faster than for-loop in CPython for tight arithmetic.
    while x <= hi:
        acc = (acc * x) % mod
        x += 1
    return acc


def _factorials_n_and_2n_mod(n: int, mod: int, workers: int) -> tuple[int, int]:
    """Return (n! mod mod, (2n)! mod mod).

    Strategy:
      - Compute A = product_{i=1..n} i   (mod mod)
      - Compute B = product_{i=n+1..2n} i (mod mod)
      - Then n! = A and (2n)! = A*B (mod mod)

    Uses multiprocessing when workers > 1.
    """

    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return 1, 1

    # Single-process path (robust and often fastest for small n).
    if workers <= 1:
        acc = 1
        fac_n = 1
        limit = 2 * n
        i = 1
        while i <= limit:
            acc = (acc * i) % mod
            if i == n:
                fac_n = acc
            i += 1
        return fac_n, acc

    # Multi-process path.
    try:
        from multiprocessing import Pool

        # Use a few more chunks than workers to improve load balancing.
        chunks = max(1, workers * 4)
        tasks_a = _split_inclusive(1, n, chunks)
        tasks_b = _split_inclusive(n + 1, 2 * n, chunks)
        tasks = tasks_a + tasks_b

        with Pool(processes=workers) as pool:
            results = pool.starmap(_prod_range_mod, [(a, b, mod) for a, b in tasks])

        a_prod = 1
        for v in results[: len(tasks_a)]:
            a_prod = (a_prod * v) % mod

        b_prod = 1
        for v in results[len(tasks_a) :]:
            b_prod = (b_prod * v) % mod

        return a_prod, (a_prod * b_prod) % mod
    except Exception:
        # If multiprocessing is unavailable or restricted, fall back gracefully.
        return _factorials_n_and_2n_mod(n, mod, workers=1)


def p_mod(n: int, mod: int = MOD, workers: int = 1) -> int:
    """Compute P(n) modulo `mod`.

    Via RSK:
      - Condition (2) means the RSK shape has at most 2 rows.
      - With 2n elements, shapes are (a, 2n-a) with a>=n.
      - Condition (1) bounds the first row: a <= n+1.
      - Therefore only shapes (n,n) and (n+1,n-1) contribute.

    The number of permutations with RSK shape λ is (f^λ)^2, where f^λ is the
    number of standard Young tableaux (SYT) of shape λ.

    For two-row shapes, hook-length gives:
        f^(n,n)         = Catalan(n) = C(2n,n)/(n+1)
        f^(n+1,n-1)     = Catalan(n) * 3n/(n+2)

    So:
        P(n) = Catalan(n)^2 * ( 1 + (3n/(n+2))^2 )
    """

    if n < 0:
        raise ValueError("n must be non-negative")

    fac_n, fac_2n = _factorials_n_and_2n_mod(n, mod, workers)

    inv_fac_n = pow(fac_n, mod - 2, mod)
    binom_2n_n = fac_2n
    binom_2n_n = (binom_2n_n * inv_fac_n) % mod
    binom_2n_n = (binom_2n_n * inv_fac_n) % mod

    catalan = (binom_2n_n * pow(n + 1, mod - 2, mod)) % mod

    t = (3 * n) % mod
    t = (t * pow(n + 2, mod - 2, mod)) % mod

    return (catalan * catalan % mod) * ((1 + t * t) % mod) % mod


def _parse_workers() -> int:
    """Pick a reasonable default for worker count.

    Override with environment variable PE916_WORKERS.
    """
    env = os.environ.get("PE916_WORKERS", "").strip()
    if env:
        try:
            v = int(env)
            return max(1, v)
        except ValueError:
            pass

    c = os.cpu_count() or 1
    # Cap to keep overhead reasonable.
    return max(1, min(8, c))


def main() -> None:
    # Test values from the problem statement.
    assert p_mod(2, MOD, workers=1) == 13
    assert p_mod(10, MOD, workers=1) == 45265702

    n = 10**8

    # Avoid multiprocessing overhead for small n, but use it for the real target.
    workers = _parse_workers()
    if workers > 1:
        # Only worth it for large n.
        workers = workers if n >= 2_000_000 else 1

    ans = p_mod(n, MOD, workers=workers)
    print(ans)


if __name__ == "__main__":
    # Allow a quick manual run for smaller n:
    #   python3 main.py 1000
    if len(sys.argv) >= 2:
        try:
            nn = int(sys.argv[1])
        except ValueError:
            raise SystemExit("Argument must be an integer n")

        ww = _parse_workers()
        if len(sys.argv) >= 3:
            try:
                ww = max(1, int(sys.argv[2]))
            except ValueError:
                raise SystemExit("Second argument must be an integer worker count")

        # Keep the statement asserts intact.
        assert p_mod(2, MOD, workers=1) == 13
        assert p_mod(10, MOD, workers=1) == 45265702

        print(p_mod(nn, MOD, workers=ww))
    else:
        main()
