#!/usr/bin/env python3
"""Project Euler 940: Two-Dimensional Recurrence

Compute S(50) mod 1123581313.

No external libraries.
"""

MOD = 1123581313

# 2x2 matrices are represented as (a,b,c,d) meaning [[a,b],[c,d]]


def mat_mul(A, B, mod=MOD):
    a, b, c, d = A
    e, f, g, h = B
    return (
        (a * e + b * g) % mod,
        (a * f + b * h) % mod,
        (c * e + d * g) % mod,
        (c * f + d * h) % mod,
    )


def mat_pow(M, exp, mod=MOD):
    # Binary exponentiation
    result = (1, 0, 0, 1)  # identity
    base = M
    e = exp
    while e > 0:
        if e & 1:
            result = mat_mul(result, base, mod)
        base = mat_mul(base, base, mod)
        e >>= 1
    return result


def mat_vec(M, x, y, mod=MOD):
    a, b, c, d = M
    return ((a * x + b * y) % mod, (c * x + d * y) % mod)


# Derived transfer matrices:
# Advancing m by 1 at fixed n maps (A(m,n), A(m,n+1)) -> (A(m+1,n), A(m+1,n+1))
MAT_M = (1, 1, 3, 2)
# Advancing n by 1 at fixed m maps (A(m,n), A(m+1,n)) -> (A(m,n+1), A(m+1,n+1))
# where A(m,n+1) = A(m+1,n) - A(m,n)
# and A(m+1,n+1) = 2*A(m+1,n) + A(m,n)
MAT_N = (MOD - 1, 1, 1, 2)  # [[-1,1],[1,2]] modulo MOD


def fibs_upto(k):
    """Return [f_0, f_1, ..., f_k] with f_0=0, f_1=1."""
    if k < 0:
        return []
    if k == 0:
        return [0]
    fib = [0, 1]
    for _ in range(2, k + 1):
        fib.append(fib[-1] + fib[-2])
    return fib


def precompute_powers(exponents, base_matrix):
    """Map each exponent e to base_matrix^e modulo MOD."""
    out = {}
    for e in exponents:
        out[e] = mat_pow(base_matrix, e)
    return out


def A_at(m, n, powM, powN, mod=MOD):
    """Compute A(m,n) modulo mod using cached matrix powers."""
    # p = (A(m,0), A(m,1)) = MAT_M^m * (0,1)
    p0, p1 = mat_vec(powM[m], 0, 1, mod)
    # A(m+1,0) = A(m,0) + A(m,1)
    q0 = (p0 + p1) % mod
    # (A(m,n), A(m+1,n)) = MAT_N^n * (A(m,0), A(m+1,0))
    r0, _ = mat_vec(powN[n], p0, q0, mod)
    return r0


def S(k, mod=MOD):
    fib = fibs_upto(k)
    idx_vals = [fib[i] for i in range(2, k + 1)]

    # Cache matrix powers for exactly the needed exponents.
    unique = sorted(set(idx_vals))
    powM = precompute_powers(unique, MAT_M)
    powN = precompute_powers(unique, MAT_N)

    total = 0
    for m in idx_vals:
        # Precompute base vector (A(m,0), A(m+1,0)) once per m.
        p0, p1 = mat_vec(powM[m], 0, 1, mod)
        q0 = (p0 + p1) % mod
        for n in idx_vals:
            a_mn, _ = mat_vec(powN[n], p0, q0, mod)
            total += a_mn
        total %= mod
    return total % mod


def _self_test():
    # Values given in the problem statement.
    assert S(3, MOD) == 30
    assert S(5, MOD) == 10396


def main():
    _self_test()
    print(S(50, MOD) % MOD)


if __name__ == "__main__":
    main()
