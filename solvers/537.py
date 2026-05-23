#!/usr/bin/env python

from math import isqrt, log

MOD = 1_004_535_809
ROOT = 3


def prime_bound(count):
    if count < 6:
        return 15
    x = count + 1
    return int(x * (log(x) + log(log(x)))) + 20


def first_primes(count):
    limit = prime_bound(count)
    while True:
        flags = bytearray(b"\x01") * (limit + 1)
        flags[0:2] = b"\x00\x00"
        for p in range(2, isqrt(limit) + 1):
            if flags[p]:
                flags[p * p : limit + 1 : p] = b"\x00" * (
                    ((limit - p * p) // p) + 1
                )
        primes = [i for i in range(2, limit + 1) if flags[i]]
        if len(primes) >= count:
            return primes
        limit *= 2


def ntt(a, invert=False):
    n = len(a)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]

    length = 2
    mod = MOD
    while length <= n:
        wlen = pow(ROOT, (mod - 1) // length, mod)
        if invert:
            wlen = pow(wlen, mod - 2, mod)
        half = length >> 1
        for i in range(0, n, length):
            w = 1
            end = i + half
            for j in range(i, end):
                u = a[j]
                v = a[j + half] * w % mod
                x = u + v
                if x >= mod:
                    x -= mod
                y = u - v
                if y < 0:
                    y += mod
                a[j] = x
                a[j + half] = y
                w = w * wlen % mod
        length <<= 1

    if invert:
        inv_n = pow(n, mod - 2, mod)
        for i, value in enumerate(a):
            a[i] = value * inv_n % mod


def multiply(a, b, degree):
    if len(a) == 1:
        factor = a[0]
        return [(factor * x) % MOD for x in b[: degree + 1]]
    if len(b) == 1:
        factor = b[0]
        return [(factor * x) % MOD for x in a[: degree + 1]]

    result_len = min(len(a) + len(b) - 1, degree + 1)
    size = 1
    full_len = len(a) + len(b) - 1
    while size < full_len:
        size <<= 1

    fa = a[:] + [0] * (size - len(a))
    fb = b[:] + [0] * (size - len(b))
    ntt(fa)
    ntt(fb)
    for i in range(size):
        fa[i] = fa[i] * fb[i] % MOD
    ntt(fa, True)
    return fa[:result_len]


def coefficient_weights(n):
    primes = first_primes(n + 1)
    weights = [1]
    for r in range(1, n + 1):
        weights.append(primes[r] - primes[r - 1])
    return weights


def T(n, k):
    result = [1]
    base = coefficient_weights(n)
    while k:
        if k & 1:
            result = multiply(result, base, n)
        k >>= 1
        if k:
            base = multiply(base, base, n)
    return result[n] if n < len(result) else 0


if __name__ == "__main__":
    assert T(3, 3) == 19
    assert T(10, 10) == 869985
    assert T(1000, 1000) == 578270566
    print(T(20_000, 20_000))
