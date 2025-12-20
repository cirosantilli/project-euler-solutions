#!/usr/bin/env python3
"""
Build 50-letter strings over {x,y,z} from a modular recurrence and count how many
ordered pairs (i, j) with 0 <= i,j < N give a neutral concatenation c(i)c(j).

Key idea (explained in README): neutrality is equivalent to the product of letters
being +1 in the quaternion group Q8 under the mapping:
    x ->  i,   y ->  j,   z -> -k
"""

import sys


def _build_mul_table_q8():
    """
    Elements are encoded as integers 0..7:
        0:  1, 1:  i, 2:  j, 3:  k,
        4: -1, 5: -i, 6: -j, 7: -k
    """
    # Multiplication among positive basis {1,i,j,k} with a separate sign table.
    base = [[0] * 4 for _ in range(4)]
    sgn = [[1] * 4 for _ in range(4)]

    for a in range(4):
        for b in range(4):
            if a == 0:
                base[a][b] = b
                sgn[a][b] = 1
            elif b == 0:
                base[a][b] = a
                sgn[a][b] = 1
            elif a == b:
                base[a][b] = 0
                sgn[a][b] = -1
            else:
                # i*j=k, j*k=i, k*i=j and anti-commutation
                if (a, b) == (1, 2):
                    base[a][b], sgn[a][b] = 3, 1
                elif (a, b) == (2, 3):
                    base[a][b], sgn[a][b] = 1, 1
                elif (a, b) == (3, 1):
                    base[a][b], sgn[a][b] = 2, 1
                elif (a, b) == (2, 1):
                    base[a][b], sgn[a][b] = 3, -1
                elif (a, b) == (3, 2):
                    base[a][b], sgn[a][b] = 1, -1
                elif (a, b) == (1, 3):
                    base[a][b], sgn[a][b] = 2, -1
                else:
                    raise RuntimeError("Unexpected basis multiplication case")

    mul = [[0] * 8 for _ in range(8)]
    for A in range(8):
        sa = 1 if A < 4 else -1
        a = A & 3
        for B in range(8):
            sb = 1 if B < 4 else -1
            b = B & 3
            s = sa * sb * sgn[a][b]
            c = base[a][b]
            mul[A][B] = c if s == 1 else (c ^ 4)
    return mul


# Precompute Q8 tables once.
_MUL = _build_mul_table_q8()

# b in {0,1,2} maps to letters x,y,z, and then to Q8 elements i,j,-k:
# x -> i (1), y -> j (2), z -> -k (7)
_GEN = (1, 2, 7)

# Right-multiply table: next_val = _R[v*3 + b]
_R = [0] * (8 * 3)
for v in range(8):
    for b in range(3):
        _R[v * 3 + b] = _MUL[v][_GEN[b]]

# Inverses in Q8: inv[e] is the unique element with e*inv[e] = inv[e]*e = 1
_INV = [0] * 8
for e in range(8):
    for f in range(8):
        if _MUL[e][f] == 0 and _MUL[f][e] == 0:
            _INV[e] = f
            break


def F(N: int) -> int:
    """
    Count ordered pairs (i, j) with 0 <= i, j < N such that c(i)c(j) is neutral.
    """
    MOD = 888_888_883
    MULT = 8888
    a = 88_888_888  # a_0

    counts = [0] * 8
    R = _R
    for _ in range(N):
        v = 0  # quaternion product for this 50-letter block
        for __ in range(50):
            v = R[v * 3 + (a % 3)]
            a = (a * MULT) % MOD
        counts[v] += 1

    total = 0
    inv = _INV
    for e in range(8):
        total += counts[e] * counts[inv[e]]
    return total


def main() -> None:
    # Tests from the statement
    assert F(10) == 13
    assert F(100) == 1224

    # Required output
    sys.stdout.write(str(F(1_000_000)) + "\n")


if __name__ == "__main__":
    main()
