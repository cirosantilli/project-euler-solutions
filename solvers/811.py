#!/usr/bin/env python3
"""
Project Euler 811 - Bitwise Recursion

We compute H(t, r) = A((2^t + 1)^r) modulo 1_000_062_031.

No third-party libraries are used.
"""

MOD = 1_000_062_031


def b(n: int) -> int:
    """Largest power of 2 dividing n (n > 0)."""
    return n & -n


def max_binom_bitlen(r: int) -> int:
    """Max bit-length among binomial coefficients C(r, k) for k=0..r."""
    c = 1
    mx = 1
    for k in range(r + 1):
        if c.bit_length() > mx:
            mx = c.bit_length()
        if k < r:
            c = c * (r - k) // (k + 1)
    return mx


def one_positions_via_binom(t: int, r: int) -> list[int]:
    """
    Positions of 1-bits in (2^t + 1)^r, assuming blocks do not overlap:
      (1 + 2^t)^r = sum_{k=0..r} C(r,k) * 2^(k*t)
    If t is at least the maximum bit-length of C(r,k), these shifted blocks are disjoint,
    so the binary representation is just the union of the bits of each C(r,k), shifted by k*t.
    """
    pos: list[int] = []
    c = 1  # C(r,0)
    for k in range(r + 1):
        x = c
        # extract set bits from low to high; yields increasing bit indices
        while x:
            lsb = x & -x
            bit = lsb.bit_length() - 1
            pos.append(k * t + bit)
            x -= lsb
        if k < r:
            c = c * (r - k) // (k + 1)
    return pos


def add_shift_positions(P: list[int], t: int) -> list[int]:
    """
    Return positions of 1-bits in (X + (X << t)) where X has 1-bits at positions in P.
    P must be strictly increasing.

    This is a sparse binary addition with carry. It's used as a fallback when t is small
    and the shifted blocks can overlap.
    """
    n = len(P)
    i = j = 0
    last = -1
    carry = False
    out: list[int] = []

    while i < n or j < n or carry:
        candidates = []
        if i < n:
            candidates.append(P[i])
        if j < n:
            candidates.append(P[j] + t)
        if carry:
            candidates.append(last + 1)

        p = min(candidates)

        s = 0
        if i < n and P[i] == p:
            s += 1
            i += 1
        if j < n and P[j] + t == p:
            s += 1
            j += 1
        if carry and p == last + 1:
            s += 1
            carry = False

        if s & 1:
            out.append(p)
        carry = s >= 2
        last = p

    return out


def one_positions_power(t: int, r: int) -> list[int]:
    """Positions of 1-bits in (2^t + 1)^r."""
    if r == 0:
        return [0]  # 1
    # Use the fast binomial-block method when safe.
    if t >= max_binom_bitlen(r):
        return one_positions_via_binom(t, r)

    # Fallback: repeated sparse multiplication by (1 + 2^t): P <- P + (P << t)
    P = [0]  # start from 1
    for _ in range(r):
        P = add_shift_positions(P, t)
    return P


def A_from_positions(pos: list[int], mod: int | None) -> int:
    """
    Compute A(n) given the positions of 1-bits in n (sorted increasing).

    Let v_0 = 1, v_{k+1} = 5*v_k + 3.
    Scan the binary representation from MSB to LSB. Each time a '0' appears,
    multiply by v_{(# of 1s to its left)}.

    If mod is None, compute the exact integer (only used for tiny tests).
    Otherwise compute modulo mod.
    """
    if not pos:
        raise ValueError("empty bit list")

    m = len(pos)
    if m == 1:
        return 1 if mod is None else 1 % mod

    # Precompute v_k up to k = m-1
    v = [0] * m
    v[0] = 1 if mod is None else 1 % mod
    for k in range(1, m):
        val = 5 * v[k - 1] + 3
        v[k] = val if mod is None else val % mod

    ans = 1 if mod is None else 1 % mod
    desc = pos[::-1]  # MSB -> LSB

    # Between consecutive 1-bits, there is a run of zeros of length "gap".
    # For the gap after the i-th 1 from the MSB, the multiplier is v_{i+1}.
    for i in range(m - 1):
        gap = desc[i] - desc[i + 1] - 1
        if gap <= 0:
            continue
        base = v[i + 1]
        if mod is None:
            ans *= pow(base, gap)
        else:
            ans = (ans * pow(base, gap, mod)) % mod

    return ans


def H(t: int, r: int, mod: int) -> int:
    """Compute H(t,r) = A((2^t + 1)^r) modulo mod."""
    pos = one_positions_power(t, r)
    # Ensure increasing order (the fast method already produces increasing output).
    if any(pos[i] >= pos[i + 1] for i in range(len(pos) - 1)):
        pos = sorted(set(pos))
    return A_from_positions(pos, mod)


def A_slow(n: int) -> int:
    """
    Direct recursion from the definition. Only intended for small n (tests).
    """
    memo: dict[int, int] = {0: 1}

    def rec(x: int) -> int:
        if x in memo:
            return memo[x]
        if x & 1:
            res = rec(x >> 1)
        else:
            m = x >> 1
            res = 3 * rec(m) + 5 * rec(x - b(m))
        memo[x] = res
        return res

    return rec(n)


def main() -> None:
    # Tests derived from the problem statement:
    assert b(24) == 8
    assert A_slow(0) == 1

    # Given: H(3,2) = A(81) = 636056
    sample_pos = one_positions_power(3, 2)  # (2^3+1)^2 = 81
    assert A_from_positions(sample_pos, None) == 636056
    assert A_slow(81) == 636056

    # Target:
    t = 10**14 + 31
    r = 62
    print(H(t, r, MOD))


if __name__ == "__main__":
    main()
