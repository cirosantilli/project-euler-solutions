#!/usr/bin/env python3
"""
Project Euler 706 - 3-Like Numbers

For a positive integer n, let f(n) be the number of non-empty substrings of the
decimal representation of n that are divisible by 3.

A d-digit number is "3-like" if f(n) is divisible by 3.
Let F(d) be the count of d-digit 3-like numbers.

Compute F(10^5) modulo 1_000_000_007.
"""

MOD = 1_000_000_007


def f_of_int(n: int) -> int:
    """
    Count substrings divisible by 3 for a (small) integer n.

    Uses the standard prefix-sum mod 3 trick:
    substring i..j is divisible by 3 <=> prefix_mod[j] == prefix_mod[i-1].
    """
    s = str(n)
    cnt = [0, 0, 0]
    cnt[0] = 1  # empty prefix
    pref = 0
    total = 0
    for ch in s:
        pref = (pref + (ord(ch) - 48)) % 3
        total += cnt[pref]
        cnt[pref] += 1
    return total


# --- Matrix utilities (small fixed size, no external libraries) ---


def mat_mul(A, B):
    """Return A*B (mod MOD). A and B are square matrices (lists of lists)."""
    n = len(A)
    C = [[0] * n for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        Ci = C[i]
        for k in range(n):
            aik = Ai[k]
            if aik:
                Bk = B[k]
                for j in range(n):
                    bkj = Bk[j]
                    if bkj:
                        Ci[j] = (Ci[j] + aik * bkj) % MOD
    return C


def mat_pow(M, e):
    """Return M**e (mod MOD)."""
    n = len(M)
    R = [[0] * n for _ in range(n)]
    for i in range(n):
        R[i][i] = 1
    A = M
    while e > 0:
        if e & 1:
            R = mat_mul(A, R)
        e >>= 1
        if e:
            A = mat_mul(A, A)
    return R


def mat_vec(M, v):
    """Return M*v (mod MOD), treating v as a column vector."""
    n = len(M)
    out = [0] * n
    for i in range(n):
        s = 0
        Mi = M[i]
        for j in range(n):
            mij = Mi[j]
            if mij:
                s = (s + mij * v[j]) % MOD
        out[i] = s
    return out


# --- Core solution ---


def _idx(a, b, c, cur):
    # a,b,c,cur each in {0,1,2}; total states = 3^4 = 81
    return ((a * 3 + b) * 3 + c) * 3 + cur


def _build_step_matrix(weights):
    """
    Build the 81x81 transition matrix for one digit step.

    State = (count0 mod 3, count1 mod 3, count2 mod 3, current_prefix_mod).
    When we append a digit with residue r, the next prefix residue becomes
    next = (cur + r) mod 3, and we increment count_next.
    """
    n = 81
    M = [[0] * n for _ in range(n)]
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for cur in range(3):
                    frm = _idx(a, b, c, cur)
                    for r, w in enumerate(weights):
                        nxt = (cur + r) % 3
                        a2, b2, c2 = a, b, c
                        if nxt == 0:
                            a2 = (a2 + 1) % 3
                        elif nxt == 1:
                            b2 = (b2 + 1) % 3
                        else:
                            c2 = (c2 + 1) % 3
                        to = _idx(a2, b2, c2, nxt)
                        M[to][frm] = (M[to][frm] + w) % MOD
    return M


def F(d: int) -> int:
    """
    Return F(d) modulo MOD.

    Leading digit: 1..9 -> residues 0/1/2 each have 3 choices.
    Other digits: 0..9 -> residue counts are (4,3,3) for (0,1,2).
    """
    if d <= 0:
        return 0

    # Transition matrices
    lead = _build_step_matrix([3, 3, 3])  # first digit
    step = _build_step_matrix([4, 3, 3])  # remaining digits

    # Precompute which states are "3-like" (f divisible by 3).
    # Let counts mod 3 be (a,b,c). Each residue contributes 1 to f mod 3 iff count ≡ 2.
    # So f mod 3 is the number of components equal to 2, modulo 3.
    good = [False] * 81
    for a in range(3):
        for b in range(3):
            for c in range(3):
                k = (1 if a == 2 else 0) + (1 if b == 2 else 0) + (1 if c == 2 else 0)
                ok = k % 3 == 0  # k in {0,1,2,3} so this means k==0 or k==3
                if ok:
                    for cur in range(3):
                        good[_idx(a, b, c, cur)] = True

    # Initial state: empty prefix sum is 0, so count0 = 1.
    v = [0] * 81
    v[_idx(1, 0, 0, 0)] = 1

    # Apply first digit
    v = mat_vec(lead, v)

    # Apply remaining digits (if any) using fast exponentiation
    if d > 1:
        P = mat_pow(step, d - 1)
        v = mat_vec(P, v)

    # Sum counts of good states
    ans = 0
    for i in range(81):
        if good[i]:
            ans += v[i]
    return ans % MOD


def main():
    # Problem statement checks
    assert f_of_int(2573) == 3
    assert F(2) == 30
    assert F(6) == 290898

    print(F(100_000) % MOD)


if __name__ == "__main__":
    main()
