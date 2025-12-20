#!/usr/bin/env python3
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


def coeff_phi6_power(k: int, m: int, mod: int = MOD, block: int = 200_000) -> int:
    """
    Compute D = [z^m] (z^2 - z + 1)^k  (mod mod), for 0 <= m <= 2k.

    Write z^2 - z + 1 = (1 + z^3)/(1 + z), then:
      (z^2 - z + 1)^k = (1 + z^3)^k * (1 + z)^(-k)

    Coefficient identity (finite sum):
      D = Σ_{j=0..⌊m/3⌋} (-1)^{m-3j} * C(k,j) * C(k + (m-3j) - 1, m-3j)

    We evaluate from j = ⌊m/3⌋ down to 0 using a rational ratio between
    consecutive terms, batching modular inverses.
    """
    if m < 0 or m > 2 * k:
        return 0

    J = m // 3
    r0 = m - 3 * J  # 0,1,2

    # term at j=J has r=r0, so the second binomial is tiny.
    term = binom_mod(k, J, mod, block)

    if r0 == 1:
        term = (term * (k % mod)) % mod
    elif r0 == 2:
        term = (term * (k % mod)) % mod
        term = (term * ((k + 1) % mod)) % mod
        term = (term * modinv(2, mod)) % mod

    # multiply by (-1)^r0
    if r0 & 1:
        term = (-term) % mod

    total = term

    j = J
    r = r0  # r associated with current term T(j)

    neg1 = mod - 1

    while j > 0:
        t = block if j >= block else j

        dens = [0] * t
        nums = [0] * t

        # For step i=0..t-1, we update from T(j-i) -> T(j-i-1).
        # Let j_cur = j - i, r_cur = r + 3*i.
        # Ratio:
        #   T(j_cur-1)/T(j_cur) =
        #     - j_cur/(k-j_cur+1) * (k+r_cur)(k+r_cur+1)(k+r_cur+2) / ((r_cur+1)(r_cur+2)(r_cur+3))
        for i in range(t):
            j_cur = j - i
            r_cur = r + 3 * i

            num = j_cur % mod
            num = (num * ((k + r_cur) % mod)) % mod
            num = (num * ((k + r_cur + 1) % mod)) % mod
            num = (num * ((k + r_cur + 2) % mod)) % mod
            nums[i] = num

            den = (k - j_cur + 1) % mod
            den = (den * ((r_cur + 1) % mod)) % mod
            den = (den * ((r_cur + 2) % mod)) % mod
            den = (den * ((r_cur + 3) % mod)) % mod
            dens[i] = den

        inv_dens = _invert_list(dens, mod)

        for i in range(t):
            term = (term * neg1) % mod
            term = (term * nums[i]) % mod
            term = (term * inv_dens[i]) % mod
            total += term
            if total >= mod:
                total -= mod

        j -= t
        r += 3 * t

    return total % mod


def amidakuji_count_mod(m: int, n: int, mod: int = MOD) -> int:
    """
    Return a(m,n) modulo `mod` (prime).

    Using S3 character decomposition:
      a(m,n) = ( C(m+n,m) + 2*D ) / 3  (mod)
    where D = [z^m] (z^2 - z + 1)^{(m+n)/2} and a(m,n)=0 if m+n is odd.
    """
    L = m + n
    if L & 1:
        return 0
    k = L // 2

    C = binom_mod(L, m, mod)
    D = coeff_phi6_power(k, m, mod)
    inv3 = modinv(3, mod)
    return ((C + 2 * D) % mod) * inv3 % mod


def _self_test():
    # test values from the problem statement
    assert amidakuji_count_mod(3, 3, MOD) == 2
    assert amidakuji_count_mod(123, 321, MOD) == 172633303


def main():
    _self_test()
    print(amidakuji_count_mod(123456789, 987654321, MOD))


if __name__ == "__main__":
    main()
