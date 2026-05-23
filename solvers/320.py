#!/usr/bin/env python

K = 1_234_567_890
MOD = 10**18


def build_spf(limit):
    spf = [0] * (limit + 1)
    for i in range(2, limit + 1):
        if spf[i] == 0:
            spf[i] = i
            if i * i <= limit:
                for j in range(i * i, limit + 1, i):
                    if spf[j] == 0:
                        spf[j] = i
    return spf


def factorial_valuation(n, p):
    total = 0
    while n:
        n //= p
        total += n
    return total


def inverse_factorial_valuation(p, target, lower_bound):
    if factorial_valuation(lower_bound, p) >= target:
        return lower_bound

    lo = lower_bound
    step = p
    hi = lo + step
    while factorial_valuation(hi, p) < target:
        lo = hi
        step <<= 1
        hi += step

    while hi - lo > 1:
        mid = (lo + hi) >> 1
        if factorial_valuation(mid, p) >= target:
            hi = mid
        else:
            lo = mid
    return hi


def compute(limit=1_000_000):
    spf = build_spf(limit)
    targets = [0] * (limit + 1)
    witnesses = [0] * (limit + 1)
    current = 0
    total = 0

    for i in range(2, limit + 1):
        n = i
        while n > 1:
            p = spf[n]
            exponent = 0
            while n % p == 0:
                n //= p
                exponent += 1

            target = targets[p] + K * exponent
            targets[p] = target
            if factorial_valuation(current, p) >= target:
                continue

            lower = target * (p - 1)
            if witnesses[p] > lower:
                lower = witnesses[p]
            witness = inverse_factorial_valuation(p, target, lower)
            witnesses[p] = witness
            if witness > current:
                current = witness

        if i >= 10:
            total = (total + current) % MOD

    return total


if __name__ == "__main__":
    assert compute(1000) == 614_538_266_565_663
    print(compute())
