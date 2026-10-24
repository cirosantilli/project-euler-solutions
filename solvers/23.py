#!/usr/bin/env python

def sum_proper_divisors_sieve(limit: int) -> list[int]:
    """
    sums[n] = sum of proper divisors of n, for 0 <= n <= limit
    """
    sums = [0] * (limit + 1)
    # add d to all multiples of d greater than d itself
    for d in range(1, limit // 2 + 1):
        for m in range(d * 2, limit + 1, d):
            sums[m] += d
    return sums


def solve(limit: int = 28123) -> int:
    """
    @formalsw: Py from project_euler_statements.p23 import naive
    @formalsw: Returns
    @formalsw: Equiv(naive)
    """
    if limit < 0:
        return 0
    sums = sum_proper_divisors_sieve(limit)
    abundant = [n for n in range(12, limit + 1) if sums[n] > n]

    # can[x] == 1 iff x can be written as sum of two abundant numbers
    can = bytearray(limit + 1)
    abund = abundant
    lim = limit
    L = len(abund)

    for i in range(L):
        a = abund[i]
        for j in range(i, L):
            s = a + abund[j]
            if s > lim:
                break
            can[s] = 1

    total = 0
    for n in range(1, lim + 1):
        if not can[n]:
            total += n
    return total


if __name__ == "__main__":
    assert sum_proper_divisors_sieve(28)[28] == 28
    assert sum_proper_divisors_sieve(12)[12] > 12
    assert solve(24) == 276
    print(solve())
