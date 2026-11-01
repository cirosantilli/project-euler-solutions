#!/usr/bin/env python


def sieve_primes(limit: int) -> list[int]:
    """
    Return the primes at most limit.
    """
    if limit < 2:
        return []
    is_prime = bytearray(b"\x01") * (limit + 1)
    is_prime[0:2] = b"\x00\x00"
    p = 2
    while p * p <= limit:
        if is_prime[p]:
            step = p
            start = p * p
            is_prime[start : limit + 1 : step] = b"\x00" * (
                ((limit - start) // step) + 1
            )
        p += 1
    return [i for i in range(2, limit + 1) if is_prime[i]]


def count_prime_summations(n: int, primes: list[int]) -> int:
    """
    Number of unordered ways to write n as a sum of primes.
    Standard coin-change DP: each prime can be used unlimited times, order doesn't matter.
    """
    ways = [0] * (n + 1)
    ways[0] = 1
    for p in primes:
        if p > n:
            break
        for s in range(p, n + 1):
            ways[s] += ways[s - p]
    return ways[n]


def solve(target: int) -> int:
    """
    @formalsw: Py from project_euler_statements.p77 import naive
    @formalsw: Returns
    @formalsw: Equiv(naive)
    """
    n = 2
    while True:
        primes = sieve_primes(n)
        cnt = count_prime_summations(n, primes)
        if cnt > target:
            return n
        n += 1


if __name__ == "__main__":
    # Example from the statement: 10 has exactly 5 prime-sum partitions
    assert count_prime_summations(10, sieve_primes(10)) == 5

    print(solve(5000))
