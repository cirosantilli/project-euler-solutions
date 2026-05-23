#!/usr/bin/env python
"""
Project Euler 837 — Amidakuji

Counts 3-line Amidakuji diagrams with:
  - m rungs between A–B (swap lines 1 and 2)
  - n rungs between B–C (swap lines 2 and 3)
whose induced permutation is the identity.

No external libraries.
"""

MOD = 1234567891


def modinv(a: int, mod: int = MOD) -> int:
    return pow(a % mod, mod - 2, mod)


def _invert_consecutive(start: int, length: int, mod: int) -> list[int]:
    """
    Return inverses of start, start+1, ..., start+length-1 modulo prime `mod`,
    using one modular exponentiation and O(length) multiplications.
    """
    if length <= 0:
        return []
    pref = [0] * length
    acc = 1
    s = start
    for i in range(length):
        acc = (acc * (s + i)) % mod
        pref[i] = acc
    inv_acc = pow(pref[-1], mod - 2, mod)
    out = [0] * length
    for i in range(length - 1, -1, -1):
        prev = pref[i - 1] if i else 1
        out[i] = (inv_acc * prev) % mod
        inv_acc = (inv_acc * (s + i)) % mod
    return out


def _invert_list(vals: list[int], mod: int) -> list[int]:
    """
    Return inverses of all nonzero `vals` modulo prime `mod`
    using one modular exponentiation and O(n) multiplications.
    """
    n = len(vals)
    if n == 0:
        return []
    pref = [0] * n
    acc = 1
    for i, v in enumerate(vals):
        acc = (acc * v) % mod
        pref[i] = acc
    inv_acc = pow(pref[-1], mod - 2, mod)
    out = [0] * n
    for i in range(n - 1, -1, -1):
        prev = pref[i - 1] if i else 1
        out[i] = (inv_acc * prev) % mod
        inv_acc = (inv_acc * vals[i]) % mod
    return out


def binom_mod(n: int, k: int, mod: int = MOD, block: int = 200_000) -> int:
    """
    Compute C(n,k) mod prime `mod` for 0 <= k <= n < mod, without factorial tables.

    Uses the product formula:
        C(n,k) = Π_{i=1..k} (n-k+i) / i (mod)
    and does denominator inversions in blocks (batch inversion).
    """
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    if k == 0:
        return 1

    base = n - k
    res = 1
    start = 1
    while start <= k:
        length = min(block, k - start + 1)
        invs = _invert_consecutive(start, length, mod)
        # multiply corresponding numerator factors
        b = base + start
        for i in range(length):
            res = (res * (b + i)) % mod
            res = (res * invs[i]) % mod
        start += length
    return res


def amidakuji_count_mod(m: int, n: int, mod: int = MOD) -> int:
    """
    Return a(m,n) modulo `mod` (prime).

    Pair adjacent letters. If k pairs are mixed (AB or BA), then
      T_k = t! / (((m-k)/2)! ((n-k)/2)! k!)
    chooses the pair layout, and
      R_k = (2^k + 2*(-1)^k) / 3
    chooses mixed orientations whose product in the order-3 subgroup is identity.
    """
    L = m + n
    if L & 1:
        return 0

    t = L // 2
    k = m & 1
    limit = min(m, n)

    if k == 0:
        layout = binom_mod(t, m // 2, mod)
    else:
        layout = (t % mod) * binom_mod(t - 1, (m - 1) // 2, mod) % mod

    inv3 = modinv(3, mod)
    pow2 = 1 if k == 0 else 2
    sign = mod - 1 if k else 1

    total = 0
    block = 200_000
    while k <= limit:
        steps = min(block, ((limit - k) // 2) + 1)
        nums = [0] * steps
        dens = [0] * steps

        kk = k
        for i in range(steps):
            nums[i] = ((m - kk) % mod) * ((n - kk) % mod) % mod
            dens[i] = (4 * (kk + 1) % mod) * ((kk + 2) % mod) % mod
            kk += 2

        inv_dens = _invert_list(dens, mod)

        for i in range(steps):
            orientations = (pow2 + 2 * sign) % mod * inv3 % mod
            total = (total + layout * orientations) % mod
            layout = layout * nums[i] % mod * inv_dens[i] % mod
            pow2 = pow2 * 4 % mod

        k += 2 * steps

    return total


def _self_test():
    # test values from the problem statement
    assert amidakuji_count_mod(3, 3, MOD) == 2
    assert amidakuji_count_mod(123, 321, MOD) == 172633303


def main():
    _self_test()
    print(amidakuji_count_mod(123456789, 987654321, MOD))


if __name__ == "__main__":
    main()
