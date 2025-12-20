#!/usr/bin/env python3
"""
Project Euler 803: Pseudorandom Sequence

Requirements satisfied:
- Pure Python, no external libraries.
- Asserts included for all test values given in the statement.
- The final answer is NOT embedded anywhere; it is computed and printed.
"""

A = 25214903917
C = 11

MOD48 = 1 << 48
MASK48 = MOD48 - 1

MOD24 = 1 << 24
MASK24 = MOD24 - 1

MOD18 = 1 << 18
MASK18 = MOD18 - 1

INV9_MOD13 = 3  # 9 * 3 ≡ 1 (mod 13)


# ----------------------------
# Letter mapping
# ----------------------------


def ch_to_val(ch: str) -> int:
    o = ord(ch)
    if 97 <= o <= 122:  # a-z
        return o - 97
    return o - 65 + 26  # A-Z


def val_to_ch(v: int) -> str:
    if v < 26:
        return chr(97 + v)
    return chr(65 + (v - 26))


# ----------------------------
# Rand48 core
# ----------------------------


def step48(x: int) -> int:
    return (A * x + C) & MASK48


def b_from_a(a: int) -> int:
    return (a >> 16) % 52


def prefix_from_seed(a0: int, length: int) -> str:
    a = a0 & MASK48
    out = []
    for _ in range(length):
        out.append(val_to_ch(b_from_a(a)))
        a = step48(a)
    return "".join(out)


def first_occurrence_bruteforce(a0: int, needle: str, limit: int) -> int:
    """
    Only used for the statement's small example (where the first occurrence is known to be 100).
    """
    L = len(needle)
    a = a0 & MASK48
    window = []
    for i in range(limit):
        window.append(val_to_ch(b_from_a(a)))
        if len(window) > L:
            window.pop(0)
        if len(window) == L and "".join(window) == needle:
            return i - L + 1
        a = step48(a)
    return -1


# ----------------------------
# Solving for seeds that match a given prefix
#
# Split 48-bit state into:
#   a = x + 2^24 * y   with x,y in [0, 2^24)
#
# Then b depends on:
#   b ≡ (x >> 16) + 48*y   (mod 52)
#
# Further split x into:
#   x = u + 2^18 * w   with u in [0, 2^18), w in [0, 64)
#
# Then:
#   (x >> 16) = (u >> 16) + 4*w
# and the mod-4 constraint on b depends only on (u >> 16).
# ----------------------------


def u0_candidates(pattern_vals):
    """Brute-force u0 in [0,2^18) that satisfies the mod-4 constraint for the whole pattern."""
    need = [v & 3 for v in pattern_vals]
    L = len(need)
    res = []
    for u0 in range(MOD18):
        u = u0
        ok = True
        for i in range(L):
            if ((u >> 16) & 3) != need[i]:
                ok = False
                break
            u = (A * u + C) & MASK18
        if ok:
            res.append(u0)
    return res


def solve_y0_for_residues(carries24, residues13):
    """
    Given:
      y_{n+1} = (A*y_n + carries24[n]) mod 2^24
      y_n mod 13 must equal residues13[n]
    Return all y0 in [0, 2^24) satisfying the constraints.
    Strategy: enumerate y0 = r0 + 13*k and prefilter using y1,y2 residues.
    """
    L = len(residues13)
    if L == 0:
        return []

    r0 = residues13[0]

    if L == 1:
        # Any y0 with correct residue works
        return list(range(r0, MOD24, 13))

    r1 = residues13[1]
    if L == 2:
        sols = []
        y0 = r0
        y1 = (A * y0 + carries24[0]) & MASK24
        delta1 = (13 * (A & MASK24)) & MASK24  # if y0 += 13 then y1 += 13*A mod 2^24
        for y0 in range(r0, MOD24, 13):
            if (y1 % 13) == r1:
                sols.append(y0)
            y1 = (y1 + delta1) & MASK24
        return sols

    # L >= 3: precheck y1 and y2 (fast, keeps scan light)
    r2 = residues13[2]

    y0 = r0
    y1 = (A * y0 + carries24[0]) & MASK24
    y2 = (A * y1 + carries24[1]) & MASK24

    a24 = A & MASK24
    delta1 = (13 * a24) & MASK24
    delta2 = (delta1 * a24) & MASK24  # 13*A^2 mod 2^24

    sols = []
    for y0 in range(r0, MOD24, 13):
        if (y1 % 13) == r1 and (y2 % 13) == r2:
            # full verify from y2 onward
            y = y2
            ok = True
            for i in range(2, L - 1):
                y = (A * y + carries24[i]) & MASK24
                if (y % 13) != residues13[i + 1]:
                    ok = False
                    break
            if ok:
                sols.append(y0)

        y1 = (y1 + delta1) & MASK24
        y2 = (y2 + delta2) & MASK24

    return sols


def solve_states_for_pattern(pattern: str):
    """
    Return all 48-bit seeds a0 such that the output string starts with 'pattern'.
    """
    vals = [ch_to_val(c) for c in pattern]
    L = len(vals)

    us = u0_candidates(vals)
    states = []

    for u0 in us:
        # Precompute u_n and k_n where:
        #   A*u_n + C = k_n*2^18 + u_{n+1}
        u = u0
        u_list = [0] * L
        k_list = [0] * (L - 1)
        for i in range(L):
            u_list[i] = u
            nxt = A * u + C
            if i < L - 1:
                k_list[i] = nxt >> 18
            u = nxt & MASK18

        # Enumerate w0 (6 bits)
        for w0 in range(64):
            w = w0
            carries24 = [0] * (L - 1)
            t_list = [0] * L  # t_n = x_n >> 16

            for i in range(L):
                t_list[i] = ((u_list[i] >> 16) & 3) + (w << 2)  # (u>>16) + 4*w
                if i < L - 1:
                    # carry for y update = floor((A*x + C)/2^24)
                    carries24[i] = (k_list[i] + A * w) >> 6
                    # w_{i+1} = (A*w_i + k_i) mod 64
                    w = (A * w + k_list[i]) & 63

            # Each character gives y_n mod 13 from:
            #   48*y_n + t_n ≡ v_n  (mod 52)
            # Reduce mod 13: 9*y_n + t_n ≡ v_n (mod 13)
            residues13 = [0] * L
            for i in range(L):
                residues13[i] = (INV9_MOD13 * ((vals[i] - (t_list[i] % 13)) % 13)) % 13

            # Solve y0 values, then build full 48-bit seed
            for y0 in solve_y0_for_residues(carries24, residues13):
                x0 = u0 + (w0 << 18)
                a0 = x0 + (y0 << 24)
                # final exact check
                if prefix_from_seed(a0, L) == pattern:
                    states.append(a0)

    return states


def find_unique_seed_for_prefix(prefix: str) -> int:
    seeds = solve_states_for_pattern(prefix)
    if len(seeds) != 1:
        raise RuntimeError(
            "Expected a unique seed for %r, got %d" % (prefix, len(seeds))
        )
    return seeds[0]


# ----------------------------
# Fast index lookup (2-adic discrete log)
#
# With a0 fixed, define:
#   K = a1 - a0  (odd => invertible mod 2^48)
#   Y_0 = 0
#   Y_{n+1} = A*Y_n + 1  (mod 2^48)
# Then:
#   a_n = a0 + K*Y_n  (mod 2^48)
# and
#   A^n ≡ (A-1)*Y_n + 1 (mod 2^48)
#
# Since A ≡ 1 (mod 4), it generates the cyclic subgroup of units ≡ 1 (mod 4),
# whose order is 2^46. We compute n mod 2^46 bit-by-bit, then test the 4 lifts.
# ----------------------------

_ORDER_EXP = 46
_G_ORDER2 = pow(A, 1 << (_ORDER_EXP - 1), MOD48)
_INV_POWS_2I = [pow(pow(A, 1 << i, MOD48), -1, MOD48) for i in range(_ORDER_EXP)]


def dlog_pow2_base_A(h: int) -> int:
    """
    Return x (mod 2^46) such that A^x ≡ h (mod 2^48), assuming h lies in <A>.
    """
    x = 0
    cur = h % MOD48
    for i in range(_ORDER_EXP):
        e = 1 << (_ORDER_EXP - 1 - i)
        t = pow(cur, e, MOD48)
        if t == _G_ORDER2:
            x |= 1 << i
            cur = (cur * _INV_POWS_2I[i]) % MOD48
        elif t != 1:
            raise RuntimeError("Discrete log failed: element not in expected subgroup")
    return x


def powA_sumY(n: int):
    """
    Compute (A^n mod 2^48, Y_n) for:
      Y_0=0, Y_{k+1}=A*Y_k+1.
    Uses binary exponentiation with a companion sum accumulator.
    """
    if n == 0:
        return 1, 0

    powv = 1
    sumv = 0
    for bit in bin(n)[2:]:
        # double
        sumv = (sumv * (1 + powv)) & MASK48
        powv = (powv * powv) & MASK48
        if bit == "1":
            # add one
            sumv = (sumv + powv) & MASK48
            powv = (powv * A) & MASK48
    return powv, sumv


def index_of_state(a0: int, target: int) -> int:
    """
    Find n such that the generator starting at a0 reaches 'target' at step n.
    """
    a0 &= MASK48
    target &= MASK48
    if target == a0:
        return 0

    a1 = step48(a0)
    K = (a1 - a0) & MASK48
    invK = pow(K, -1, MOD48)

    y_target = ((target - a0) & MASK48) * invK % MOD48
    h = ((A - 1) * y_target + 1) & MASK48  # equals A^n

    n0 = dlog_pow2_base_A(h)  # n mod 2^46

    stepN = 1 << _ORDER_EXP
    for t in range(4):
        n = n0 + t * stepN
        _, y = powA_sumY(n)
        if y == y_target:
            return n

    raise RuntimeError("Failed to lift discrete log to the correct n")


# ----------------------------
# Main
# ----------------------------


def main():
    # Statement checks
    assert prefix_from_seed(123456, 9) == "bQYicNGCY"
    assert first_occurrence_bruteforce(123456, "RxqLBfWzv", 2000) == 100
    assert find_unique_seed_for_prefix("EULERcats") == 78580612777175

    # Actual task
    seed = find_unique_seed_for_prefix("PuzzleOne")
    lucky_states = solve_states_for_pattern("LuckyText")

    best = None
    for st in lucky_states:
        n = index_of_state(seed, st)
        if best is None or n < best:
            best = n

    # Do not assert / embed the final answer; print only.
    print(best)


if __name__ == "__main__":
    main()
