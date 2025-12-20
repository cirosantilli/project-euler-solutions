#!/usr/bin/env python3
"""Project Euler 419: Look and Say Sequence

Compute A(n), B(n), C(n) = counts of digits 1,2,3 in the n-th term (starting at 1).
Answer is required modulo 2^30.

This solution builds Conway's element-decay system *from scratch* using a parsimonious
splitting oracle (spl0) for the look-and-say splitting relation, then uses matrix
exponentiation.

No external libraries.
"""

from __future__ import annotations

MOD_MASK = (1 << 30) - 1


def say(s: str) -> str:
    """One look-and-say step on a digit string."""
    if not s:
        return ""
    out_parts = []
    n = len(s)
    i = 0
    while i < n:
        ch = s[i]
        j = i + 1
        while j < n and s[j] == ch:
            j += 1
        out_parts.append(str(j - i))
        out_parts.append(ch)
        i = j
    return "".join(out_parts)


# --- Conway/Wilkins splitting oracle (Appendix A of Watkins' paper), specialized to digits 1..3 ---


def _spl00_at(s: str, j: int) -> bool:
    """spl00 on suffix s[j:], implemented without slicing."""
    n = len(s)
    # spl00 (1:1:1:_) = True
    if j + 2 < n and s[j] == "1" and s[j + 1] == "1" and s[j + 2] == "1":
        return True

    # spl00 (1:[]) = False
    if j == n - 1 and s[j] == "1":
        return False

    # spl00 (1:1:_) = False
    if j + 1 < n and s[j] == "1" and s[j + 1] == "1":
        return False

    # spl00 (1:2:2:_) = False
    if j + 2 < n and s[j] == "1" and s[j + 1] == "2" and s[j + 2] == "2":
        return False

    # spl00 (1:3:3:_) = False
    if j + 2 < n and s[j] == "1" and s[j + 1] == "3" and s[j + 2] == "3":
        return False

    # spl00 (2:_) = False
    if j < n and s[j] == "2":
        return False

    # spl00 (3:1:1:1:_) = False
    if (
        j + 3 < n
        and s[j] == "3"
        and s[j + 1] == "1"
        and s[j + 2] == "1"
        and s[j + 3] == "1"
    ):
        return False

    # spl00 (3:2:2:2:_) = False
    if (
        j + 3 < n
        and s[j] == "3"
        and s[j + 1] == "2"
        and s[j + 2] == "2"
        and s[j + 3] == "2"
    ):
        return False

    # spl00 (3:3:_) = False
    if j + 1 < n and s[j] == "3" and s[j + 1] == "3":
        return False

    # default spl00 = True
    return True


def _spl0_at(s: str, i: int) -> bool:
    """spl0 on suffix s[i:], implemented without slicing."""
    n = len(s)
    ch = s[i]

    # spl0(1:[ ]) = True
    if ch == "1" and i == n - 1:
        return True

    # spl0(1:2:2:xs) = spl00 xs
    if ch == "1" and i + 2 < n and s[i + 1] == "2" and s[i + 2] == "2":
        return _spl00_at(s, i + 3)

    # spl0(2:xs) = spl00 xs
    if ch == "2":
        return _spl00_at(s, i + 1)

    # spl0(3:[ ]) = True
    if ch == "3" and i == n - 1:
        return True

    # spl0(3:2:2:xs) = spl00 xs
    if ch == "3" and i + 2 < n and s[i + 1] == "2" and s[i + 2] == "2":
        return _spl00_at(s, i + 3)

    # remaining 4-cases don't occur for this problem; default spl0 = False
    return False


def split_elements(s: str) -> list[str]:
    """Factor s into Conway 'elements' using the parsimonious splitting oracle."""
    if not s:
        return []
    parts: list[str] = []
    start = 0
    n = len(s)
    for i in range(n):
        if _spl0_at(s, i):
            parts.append(s[start : i + 1])
            start = i + 1
    if start < n:
        parts.append(s[start:])
    return parts


# --- Linear algebra mod 2^30 ---


def mat_mul(A: list[list[int]], B: list[list[int]]) -> list[list[int]]:
    """Dense matrix multiply mod 2^30."""
    m = len(A)
    res = [[0] * m for _ in range(m)]
    for i in range(m):
        Ai = A[i]
        Ri = res[i]
        for k in range(m):
            a = Ai[k]
            if a:
                Bk = B[k]
                # accumulate row
                for j in range(m):
                    Ri[j] = (Ri[j] + a * Bk[j]) & MOD_MASK
    return res


def vec_mul(v: list[int], M: list[list[int]]) -> list[int]:
    """Row-vector times matrix mod 2^30."""
    m = len(v)
    out = [0] * m
    for i, a in enumerate(v):
        if a:
            row = M[i]
            for j in range(m):
                out[j] = (out[j] + a * row[j]) & MOD_MASK
    return out


def vec_mul_pow(v: list[int], M: list[list[int]], exp: int) -> list[int]:
    """Compute v * (M ** exp) mod 2^30."""
    while exp > 0:
        if exp & 1:
            v = vec_mul(v, M)
        exp >>= 1
        if exp:
            M = mat_mul(M, M)
    return v


# --- Build decay system from scratch ---


def build_decay_system(seed_term: str):
    """Return (elements_list, decay_matrix, digit_count_arrays, seed_vector)."""
    seed_elements = split_elements(seed_term)

    elems: list[str] = []
    idx: dict[str, int] = {}

    def _add(e: str) -> int:
        j = idx.get(e)
        if j is None:
            j = len(elems)
            idx[e] = j
            elems.append(e)
        return j

    # start with all elements in the seed decomposition
    for e in seed_elements:
        _add(e)

    # closure under decay
    p = 0
    while p < len(elems):
        e = elems[p]
        d = split_elements(say(e))
        for child in d:
            _add(child)
        p += 1

    m = len(elems)

    # build decay matrix
    M = [[0] * m for _ in range(m)]
    for i, e in enumerate(elems):
        d = split_elements(say(e))
        row = M[i]
        for child in d:
            row[idx[child]] += 1

    # digit counts per element
    ones = [0] * m
    twos = [0] * m
    threes = [0] * m
    for i, e in enumerate(elems):
        ones[i] = e.count("1")
        twos[i] = e.count("2")
        threes[i] = e.count("3")

    # seed vector
    v0 = [0] * m
    for e in seed_elements:
        v0[idx[e]] += 1

    return elems, M, ones, twos, threes, v0


def counts_in_term(term: str) -> tuple[int, int, int]:
    return term.count("1"), term.count("2"), term.count("3")


def solve(n: int) -> tuple[int, int, int]:
    if n < 1:
        raise ValueError("n must be >= 1")

    # direct simulation is cheap for small n (and used for the provided check)
    term = "1"
    for _ in range(1, min(n, 40)):
        term = say(term)

    if n <= 40:
        return counts_in_term(term)

    # term is now the 40th term
    a40, b40, c40 = counts_in_term(term)
    # test value from the problem statement
    assert (a40, b40, c40) == (31254, 20259, 11625)

    # build system from the mature 40th term, then jump to n
    _, M, ones, twos, threes, v = build_decay_system(term)

    # exponentiate from term 40 to term n
    v = vec_mul_pow(v, M, n - 40)

    A = 0
    B = 0
    C = 0
    for i, cnt in enumerate(v):
        if cnt:
            A = (A + cnt * ones[i]) & MOD_MASK
            B = (B + cnt * twos[i]) & MOD_MASK
            C = (C + cnt * threes[i]) & MOD_MASK
    return A, B, C


def main() -> None:
    n = 10**12
    A, B, C = solve(n)
    print(f"{A},{B},{C}")


if __name__ == "__main__":
    main()
