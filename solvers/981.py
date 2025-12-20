#!/usr/bin/env python3
# No external libraries; single-threaded solution.

import sys
import math
from array import array


MOD = 888_888_883


def _qbinom_minus1_int(n: int, k: int) -> int:
    """Gaussian binomial C(n,k)_q evaluated at q=-1, as an integer."""
    # Known closed form:
    #   C(n,k)_{-1} = 0                if n even and k odd
    #             = binom(n//2, k//2)  otherwise (floors).
    if (n & 1) == 0 and (k & 1) == 1:
        return 0
    return math.comb(n >> 1, k >> 1)


def N_exact(X: int, Y: int, Z: int) -> int:
    """Exact N(X,Y,Z) for small inputs (used only for statement asserts)."""
    if (X & 1) != (Y & 1) or (Y & 1) != (Z & 1):
        return 0
    n = X + Y + Z
    total = math.factorial(n) // (
        math.factorial(X) * math.factorial(Y) * math.factorial(Z)
    )
    diff = _qbinom_minus1_int(n, X) * _qbinom_minus1_int(n - X, Y)  # E - O
    # Canonical-product sign: +1 if (X//2+Y//2+Z//2) even else -1
    sign = 1 if (((X >> 1) + (Y >> 1) + (Z >> 1)) & 1) == 0 else -1
    return (total + sign * diff) // 2


def solve() -> int:
    # Tests from the statement (exact values)
    assert N_exact(2, 2, 2) == 42
    assert N_exact(8, 8, 8) == 4732773210

    cubes = [i * i * i for i in range(88)]
    max_n = 3 * cubes[-1]  # maximum X+Y+Z needed
    # max_n < MOD, so factorial values are invertible mod MOD.

    # Precompute factorials and inverse factorials modulo MOD.
    fact = array("I", [1]) * (max_n + 1)
    for i in range(1, max_n + 1):
        fact[i] = (fact[i - 1] * i) % MOD

    invfact = array("I", [1]) * (max_n + 1)
    invfact[max_n] = pow(int(fact[max_n]), -1, MOD)
    for i in range(max_n, 0, -1):
        invfact[i - 1] = (invfact[i] * i) % MOD

    inv2 = (MOD + 1) // 2  # inverse of 2 modulo any odd MOD

    # Precompute per-cube helpers for speed.
    halves = [c >> 1 for c in cubes]
    invf = [int(invfact[c]) for c in cubes]
    par = [i & 1 for i in range(88)]  # same parity as cube

    def comb_mod(n: int, k: int) -> int:
        if k < 0 or k > n:
            return 0
        return (int(fact[n]) * int(invfact[k]) % MOD) * int(invfact[n - k]) % MOD

    total_sum = 0
    fact_local = fact
    invfact_local = invfact  # keep local references

    for ai, X in enumerate(cubes):
        hx = halves[ai]
        invX = invf[ai]
        px = par[ai]
        for bj, Y in enumerate(cubes):
            hy = halves[bj]
            invY = invf[bj]
            py = par[bj]
            for ck, Z in enumerate(cubes):
                if px != py or py != par[ck]:
                    continue  # impossible to reach; hence N=0

                hz = halves[ck]
                invZ = invf[ck]

                n = X + Y + Z

                # T = multinomial(n; X,Y,Z) mod MOD
                T = int(fact_local[n])
                T = (T * invX) % MOD
                T = (T * invY) % MOD
                T = (T * invZ) % MOD

                # D = (E - O) = q-multinomial evaluated at q=-1:
                # D = C(n, X)_{-1} * C(n-X, Y)_{-1}
                if (n & 1) == 0 and (X & 1) == 1:
                    D = 0
                else:
                    D1 = comb_mod(n >> 1, hx)
                    n2 = n - X  # = Y+Z
                    if (n2 & 1) == 0 and (Y & 1) == 1:
                        D = 0
                    else:
                        D2 = comb_mod(n2 >> 1, hy)
                        D = (D1 * D2) % MOD

                # Canonical-product sign (in Q8 under x->i, y->j, z->-k):
                # p0 = +1 if (X//2+Y//2+Z//2) even else -1
                if ((hx + hy + hz) & 1) == 0:
                    Nmod = (T + D) % MOD
                else:
                    Nmod = (T - D) % MOD

                Nmod = (Nmod * inv2) % MOD
                total_sum += Nmod
                if total_sum >= MOD:
                    total_sum -= MOD

    return total_sum % MOD


def main() -> None:
    sys.stdout.write(str(solve()) + "\n")


if __name__ == "__main__":
    main()
