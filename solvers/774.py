#!/usr/bin/env python3
"""
Project Euler 774 - Conjunctive Sequences

We count sequences (a1..an) with 0 <= ai <= b and (ai & a{i+1}) != 0.
Answer requested: c(123, 123456789) mod 998244353.

No external libraries are used.
"""

MOD = 998244353

# Per-bit matrix for the predicate (x & y) == 0 in one bit:
# rows = y_bit, cols = x_bit
# allowed pairs: (y=0,x=0/1) and (y=1,x=0)
R_DISJOINT = ((1, 1), (1, 0))


class TT:
    """
    Tensor-Train / Matrix Product State representation of a length-m
    binary-indexed vector of size 2^m.

    Each core is a 3D nested list core[l][bit][r] with:
      - left rank dimension rL = len(core)
      - physical dimension = 2 (bit 0/1)
      - right rank dimension rR = len(core[0][0])
    """

    __slots__ = ("cores", "m")

    def __init__(self, cores):
        self.cores = cores
        self.m = len(cores)


def tt_all_ones(m):
    # Rank-1 tensor train for the all-ones vector on {0,1}^m.
    # Each core has shape (1,2,1) with entries 1.
    cores = []
    for _ in range(m):
        cores.append([[[1], [1]]])
    return TT(cores)


def tt_indicator_leq(b, m):
    """
    Builds an indicator vector 1_{x <= b} (for x in [0,2^m)).
    Implemented as a 2-state DFA (tight/loose) compiled into TT form.
    """
    # Bits from MSB -> LSB
    bits = [(b >> (m - 1 - i)) & 1 for i in range(m)]
    cores = []

    if m == 1:
        # Special-case: a single core with boundaries absorbed.
        bb = bits[0]
        # Transition matrices rows=prev_state, cols=next_state
        if bb == 0:
            T0 = ((1, 0), (0, 1))
            T1 = ((1, 0), (0, 0))
        else:
            T0 = ((1, 0), (1, 0))
            T1 = ((1, 0), (0, 1))
        # start = tight, end = accept both
        # core[0][x][0] = sum_{prev,nxt} start[prev]*T_x[prev][nxt]*end[nxt]
        core = [[[0], [0]]]  # (1,2,1)
        for xbit, T in enumerate((T0, T1)):
            # start is (0,1), so prev=tight=1
            val = (T[1][0] + T[1][1]) % MOD
            core[0][xbit][0] = val
        return TT([core])

    for idx, bb in enumerate(bits):
        # Per-bit transition matrices (prev_state -> next_state)
        if bb == 0:
            T0 = ((1, 0), (0, 1))  # x=0
            T1 = ((1, 0), (0, 0))  # x=1
        else:
            T0 = ((1, 0), (1, 0))  # x=0
            T1 = ((1, 0), (0, 1))  # x=1

        if idx == 0:
            # absorb start state (tight)
            core = [[[0, 0], [0, 0]]]  # (1,2,2)
            # prev = tight = 1
            for xbit, T in enumerate((T0, T1)):
                core[0][xbit][0] = T[1][0]
                core[0][xbit][1] = T[1][1]
            cores.append(core)

        elif idx == m - 1:
            # absorb end acceptance vector (1,1)
            core = [[[0], [0]] for _ in range(2)]  # (2,2,1)
            for prev in range(2):
                for xbit, T in enumerate((T0, T1)):
                    core[prev][xbit][0] = (T[prev][0] + T[prev][1]) % MOD
            cores.append(core)

        else:
            core = [[[0, 0], [0, 0]] for _ in range(2)]  # (2,2,2)
            for prev in range(2):
                for xbit, T in enumerate((T0, T1)):
                    core[prev][xbit][0] = T[prev][0]
                    core[prev][xbit][1] = T[prev][1]
            cores.append(core)

    return TT(cores)


def tt_scalar_mul(tt, c):
    c %= MOD
    cores = []
    for core in tt.cores:
        # deep copy each core
        new_core = [[row[:] for row in sl] for sl in core]
        cores.append(new_core)
    # scaling one core is enough
    core0 = cores[0]
    for l in range(len(core0)):
        for bit in range(2):
            row = core0[l][bit]
            for r in range(len(row)):
                row[r] = (row[r] * c) % MOD
    return TT(cores)


def tt_add(a, b, coef_b=1):
    """
    Returns a + coef_b * b in TT form.
    IMPORTANT: coef_b is applied only to the FIRST core of b
    (scaling a single core scales the whole tensor).
    """
    coef_b %= MOD
    m = a.m
    assert m == b.m
    cores = []

    for i in range(m):
        A = a.cores[i]
        B = b.cores[i]
        rLa, rRa = len(A), len(A[0][0])
        rLb, rRb = len(B), len(B[0][0])

        if i == 0:
            # (1,2,rRa+rRb)
            core = [[[0] * (rRa + rRb) for _ in range(2)]]
            for bit in range(2):
                core[0][bit][:rRa] = A[0][bit][:]
                core[0][bit][rRa:] = [(coef_b * v) % MOD for v in B[0][bit]]
            cores.append(core)

        elif i == m - 1:
            # (rLa+rLb,2,1)
            core = [[[0], [0]] for _ in range(rLa + rLb)]
            for l in range(rLa):
                core[l][0][0] = A[l][0][0]
                core[l][1][0] = A[l][1][0]
            for l in range(rLb):
                core[rLa + l][0][0] = B[l][0][0]
                core[rLa + l][1][0] = B[l][1][0]
            cores.append(core)

        else:
            # block diagonal
            core = [[[0] * (rRa + rRb) for _ in range(2)] for _ in range(rLa + rLb)]
            # A in top-left
            for l in range(rLa):
                core[l][0][:rRa] = A[l][0][:]
                core[l][1][:rRa] = A[l][1][:]
            # B in bottom-right
            for l in range(rLb):
                core[rLa + l][0][rRa:] = B[l][0][:]
                core[rLa + l][1][rRa:] = B[l][1][:]
            cores.append(core)

    return TT(cores)


def tt_hadamard(a, b):
    """
    Elementwise product (Hadamard). Ranks multiply.
    """
    m = a.m
    assert m == b.m
    cores = []

    for i in range(m):
        A = a.cores[i]
        B = b.cores[i]
        rLa, rRa = len(A), len(A[0][0])
        rLb, rRb = len(B), len(B[0][0])

        rL = rLa * rLb
        rR = rRa * rRb
        core = [[[0] * rR for _ in range(2)] for _ in range(rL)]

        for la in range(rLa):
            for lb in range(rLb):
                l = la * rLb + lb
                for bit in range(2):
                    Arow = A[la][bit]
                    Brow = B[lb][bit]
                    out = core[l][bit]
                    for ra in range(rRa):
                        av = Arow[ra]
                        if av:
                            base = ra * rRb
                            for rb in range(rRb):
                                out[base + rb] = (out[base + rb] + av * Brow[rb]) % MOD

        cores.append(core)

    return TT(cores)


def tt_apply_local(tt, M):
    """
    Applies a per-bit 2x2 matrix M (rows=new_bit, cols=old_bit)
    to each physical dimension. This is valid because the operator is
    a Kronecker product across bits.
    """
    m00, m01 = M[0]
    m10, m11 = M[1]
    cores = []

    for core in tt.cores:
        rL = len(core)
        rR = len(core[0][0])
        new_core = [[[0] * rR for _ in range(2)] for _ in range(rL)]

        for l in range(rL):
            o0 = core[l][0]
            o1 = core[l][1]
            n0 = new_core[l][0]
            n1 = new_core[l][1]
            for r in range(rR):
                a0 = o0[r]
                a1 = o1[r]
                n0[r] = (m00 * a0 + m01 * a1) % MOD
                n1[r] = (m10 * a0 + m11 * a1) % MOD

        cores.append(new_core)

    return TT(cores)


def tt_sum_all(tt):
    """
    Sums all entries of the represented vector (contracts with all-ones).
    """
    vec = [1]  # left boundary
    for core in tt.cores:
        rL = len(core)
        rR = len(core[0][0])
        # vec length matches rL
        new = [0] * rR
        for l in range(rL):
            vl = vec[l]
            if vl:
                c0 = core[l][0]
                c1 = core[l][1]
                for r in range(rR):
                    new[r] = (new[r] + vl * (c0[r] + c1[r])) % MOD
        vec = new
    return vec[0] % MOD


def _reduce_pivots_only(rr):
    """
    Faster elimination specialized for our compression:
      1) forward elimination to row-echelon (normalize pivots; eliminate below)
      2) eliminate above pivots only within pivot rows (not all rows)

    After this, pivot columns restricted to pivot rows become the identity,
    and rr[k][j] in pivot row k provides coefficient of basis-column k
    when expressing column j.

    Returns (pivot_cols, rank). rr is modified in-place.
    """
    mod = MOD
    powmod = pow
    m = len(rr)
    n = len(rr[0]) if m else 0

    pivots = []
    r = 0

    # Forward elimination
    for c in range(n):
        piv = None
        for i in range(r, m):
            if rr[i][c]:
                piv = i
                break
        if piv is None:
            continue

        if piv != r:
            rr[r], rr[piv] = rr[piv], rr[r]

        row = rr[r]
        inv = powmod(row[c], mod - 2, mod)

        for j in range(c, n):
            row[j] = (row[j] * inv) % mod

        for i in range(r + 1, m):
            rowi = rr[i]
            f = rowi[c]
            if f:
                for j in range(c, n):
                    rowi[j] = (rowi[j] - f * row[j]) % mod

        pivots.append(c)
        r += 1
        if r == m:
            break

    rank = r

    # Back elimination, but only among pivot rows
    for k in range(rank - 1, -1, -1):
        c = pivots[k]
        rowk = rr[k]
        for i in range(k):
            rowi = rr[i]
            f = rowi[c]
            if f:
                for j in range(c, n):
                    rowi[j] = (rowi[j] - f * rowk[j]) % mod

    return pivots, rank


def tt_reduce_left(tt):
    """
    Exact TT compression from left to right:
    For each core, we remove dependent columns in the unfolding matrix
    by Gaussian elimination over the finite field mod MOD.

    This keeps ranks from exploding while preserving exactness.
    """
    cores = [c for c in tt.cores]
    m = len(cores)

    for i in range(m - 1):
        core = cores[i]
        rL = len(core)
        rR = len(core[0][0])

        # build row copies (2*rL rows)
        rr = [None] * (2 * rL)
        for l in range(rL):
            rr[2 * l] = core[l][0][:]
            rr[2 * l + 1] = core[l][1][:]

        pivs, rank = _reduce_pivots_only(rr)

        if rank == rR:
            continue  # already full column rank

        # new core keeps only pivot columns
        new_core = [[[0] * rank for _ in range(2)] for __ in range(rL)]
        for l in range(rL):
            o0 = core[l][0]
            o1 = core[l][1]
            row0 = new_core[l][0]
            row1 = new_core[l][1]
            for k, p in enumerate(pivs):
                row0[k] = o0[p]
                row1[k] = o1[p]
        cores[i] = new_core

        # absorb coefficient matrix into next core
        nxt = cores[i + 1]
        rNext = len(nxt[0][0])
        new_nxt = [[[0] * rNext for _ in range(2)] for __ in range(rank)]

        pivot_set = set(pivs)

        # pivot columns contribute with coefficient 1
        for k, p in enumerate(pivs):
            rowp = nxt[p]
            for bit in range(2):
                dest = new_nxt[k][bit]
                src = rowp[bit]
                for t in range(rNext):
                    dest[t] = (dest[t] + src[t]) % MOD

        # non-pivot columns: coefficient = rr[k][j] on pivot row k
        for j in range(rR):
            if j in pivot_set:
                continue
            rowj = nxt[j]
            for k in range(rank):
                coeff = rr[k][j]
                if coeff:
                    for bit in range(2):
                        dest = new_nxt[k][bit]
                        src = rowj[bit]
                        for t in range(rNext):
                            dest[t] = (dest[t] + coeff * src[t]) % MOD

        cores[i + 1] = new_nxt

    return TT(cores)


def c(n, b):
    """
    Computes c(n,b) mod MOD using TT/MPS iteration:
        dp_{t+1} = 1_{x<=b} * ( (J - B) dp_t )
    where B_{y,x}=1 iff x&y==0 and J is all-ones.
    """
    m = max(1, b.bit_length())

    mask = tt_indicator_leq(b, m)
    dp = tt_reduce_left(mask)

    ones = tt_all_ones(m)

    for _ in range(n - 1):
        total = tt_sum_all(dp)
        j = tt_scalar_mul(ones, total)
        bv = tt_apply_local(dp, R_DISJOINT)
        nxt = tt_add(j, bv, coef_b=-1)  # (J - B) dp
        nxt = tt_hadamard(nxt, mask)  # project y<=b
        dp = tt_reduce_left(nxt)

    return tt_sum_all(dp)


def main():
    # Problem statement test values
    assert c(3, 4) == 18
    assert c(10, 6) == 2496120
    assert c(100, 200) == 268159379

    # Required computation
    print(c(123, 123456789) % MOD)


if __name__ == "__main__":
    main()
