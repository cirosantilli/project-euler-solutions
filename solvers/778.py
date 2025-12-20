#!/usr/bin/env python3
"""
Project Euler 778: Freshman's Product

We define the freshman's product ⊠ digitwise:
(a ⊠ b)_i = (a_i * b_i) mod 10   (no carries, missing digits are treated as 0).

Let F(R, M) be the sum of x1 ⊠ x2 ⊠ ... ⊠ xR over all sequences (x1,...,xR) with 0 <= xi <= M.
We compute F(234567, 765432) modulo 1_000_000_009.

No external libraries are used.
"""

MOD = 1_000_000_009


def freshman_product(a: int, b: int) -> int:
    """Digitwise product modulo 10 (freshman's product)."""
    res = 0
    place = 1
    while a > 0 or b > 0:
        da = a % 10
        db = b % 10
        res += ((da * db) % 10) * place
        place *= 10
        a //= 10
        b //= 10
    return res


def digit_counts_upto(n: int, pos: int) -> list:
    """
    Counts of digits 0..9 at decimal position `pos` (0 = units) among numbers 0..n inclusive.
    Numbers are considered with leading zeros, which matches the definition of ⊠.
    """
    base = 10**pos
    higher = n // (base * 10)
    cur = (n // base) % 10
    lower = n % base

    counts = [higher * base] * 10
    for d in range(cur):
        counts[d] += base
    counts[cur] += lower + 1
    return counts


def mat_mul(A: list, B: list, mod: int) -> list:
    """Multiply two 10x10 matrices modulo mod."""
    n = 10
    C = [[0] * n for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        for k in range(n):
            aik = Ai[k]
            if aik:
                Bk = B[k]
                for j in range(n):
                    C[i][j] = (C[i][j] + aik * Bk[j]) % mod
    return C


def mat_pow(A: list, exp: int, mod: int) -> list:
    """Fast exponentiation of a 10x10 matrix modulo mod."""
    n = 10
    R = [[0] * n for _ in range(n)]
    for i in range(n):
        R[i][i] = 1

    while exp:
        if exp & 1:
            R = mat_mul(R, A, mod)
        A = mat_mul(A, A, mod)
        exp >>= 1
    return R


def F(R: int, M: int, mod: int = MOD) -> int:
    """
    Compute F(R, M) modulo mod.

    For each digit position independently:
      - count how many numbers in [0..M] have digit d at that position (counts[d])
      - model the product digit as a 10-state automaton (state = current product mod 10)
      - use matrix exponentiation to apply R independent choices
    """
    max_digits = len(str(M))  # digits beyond this are always 0 for all x in [0..M]
    ans = 0
    pow10 = 1

    for pos in range(max_digits):
        counts = digit_counts_upto(M, pos)

        # Transition matrix: from state s to state t with weight = number of choices for digit d
        # such that (s * d) % 10 == t.
        A = [[0] * 10 for _ in range(10)]
        for s in range(10):
            row = A[s]
            for d in range(10):
                t = (s * d) % 10
                row[t] = (row[t] + counts[d]) % mod

        P = mat_pow(A, R, mod)

        # Starting product is 1, so we only need row 1 of P (row-vector * matrix power).
        row1 = P[1]
        digit_sum = 0
        for digit, ways in enumerate(row1):
            digit_sum = (digit_sum + digit * ways) % mod

        ans = (ans + digit_sum * pow10) % mod
        pow10 = (pow10 * 10) % mod

    return ans


def _run_tests() -> None:
    # Examples from the problem statement
    assert freshman_product(234, 765) == 480
    assert F(2, 7) == 204
    assert F(23, 76) % MOD == 5_870_548


def main() -> None:
    _run_tests()
    print(F(234_567, 765_432) % MOD)


if __name__ == "__main__":
    main()
