#!/usr/bin/env python


def distinct_prime_factor_counts(limit: int) -> list[int]:
    """
    Returns an array cnt where cnt[n] = number of distinct prime factors of n, for 0 <= n <= limit.
    Uses a sieve-like process: for each prime p, increment all multiples of p.
    """
    cnt = [0] * (limit + 1)
    for p in range(2, limit + 1):
        if cnt[p] == 0:  # p is prime
            for m in range(p, limit + 1, p):
                cnt[m] += 1
    return cnt


def solve(k: int, run_len: int, start_limit: int) -> int:
    """
    Finds the first integer n such that n..n+run_len-1 each have exactly k distinct prime factors.

    @formalsw: Requires(k >= 1 and run_len == k and start_limit >= 1)
    @formalsw: Py from project_euler_statements.p47 import naive_solver
    @formalsw: Equiv(naive_solver)
    """
    limit = start_limit
    while True:
        cnt = distinct_prime_factor_counts(limit)
        streak = 0
        for n in range(2, limit + 1):
            if cnt[n] == k:
                streak += 1
                if streak == run_len:
                    return n - run_len + 1
            else:
                streak = 0
        limit *= 2  # expand search if not found


if __name__ == "__main__":
    # Problem statement examples / sanity checks
    assert solve(2, 2, 100) == 14
    assert solve(3, 3, 2000) == 644

    # Target problem
    print(solve(4, 4, 200_000))
