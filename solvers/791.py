#!/usr/bin/env python3
"""
Project Euler 791: Average and Variance

Compute S(n): the sum of all quadruples (a,b,c,d) with 1 <= a <= b <= c <= d <= n
whose average equals exactly twice their variance.

No external libraries are used.
"""

from math import isqrt


def solve(n: int, mod: int | None = None) -> int:
    """
    Return S(n). If mod is given, returns S(n) modulo mod (using periodic reduction
    for speed); otherwise returns the exact integer.
    """
    if n <= 0:
        return 0

    N2 = 2 * n
    R = isqrt(2 * n) + 2  # safe bound for the transformed lattice variables

    # Prefix sums up to R:
    # s2[i]  = sum_{k=0..i} k^2
    # ps2[i] = sum_{t=0..i} s2[t]  (prefix-of-prefix for fast aggregated segments)
    s2 = [0] * (R + 1)
    ps2 = [0] * (R + 1)
    p1 = [0] * (R + 1)  # sum k
    p2 = [0] * (R + 1)  # sum k^2
    p3 = [0] * (R + 1)  # sum k^3

    for i in range(1, R + 1):
        ii = i * i
        s2[i] = s2[i - 1] + ii
        ps2[i] = ps2[i - 1] + s2[i]
        p1[i] = p1[i - 1] + i
        p2[i] = p2[i - 1] + ii
        p3[i] = p3[i - 1] + ii * i

    total = 0
    if mod is None:

        def add(v: int) -> None:
            nonlocal total
            total += v

    else:
        # Reduce occasionally to avoid lots of % operations.
        LIM = 1 << 63

        def add(v: int) -> None:
            nonlocal total
            total += v
            if total >= LIM:
                total %= mod

    # Small U cases are few and easiest to brute (ensures a>=1 in all edge cases).
    for U in range(0, 2):
        for V in range(0, U + 1):
            for W in range(0, V + 1):
                for sgn in (-1, 1):
                    if W == 0 and sgn == -1:
                        continue  # w=0 would be double-counted
                    u = -U
                    v = -V
                    w = sgn * W
                    m = U * U + V * V + W * W
                    if m == 0:
                        continue
                    # Reconstruct (a,b,c,d) from (u,v,w) via x = Hadamard(u,v,w)
                    a = (m + u + v + w) // 2
                    b = (m + u - v - w) // 2
                    c = (m - u + v - w) // 2
                    d = (m - u - v + w) // 2
                    if 1 <= a <= b <= c <= d <= n:
                        v_add = 2 * m
                        add(v_add if mod is None else (v_add % mod))

    # Main enumeration:
    # We use the (U,V,W,sgn) parametrization described in README.md.
    isqrt_local = isqrt
    s2_local = s2
    ps2_local = ps2
    p1_local = p1
    p2_local = p2
    p3_local = p3

    for U in range(2, R + 1):
        U2 = U * U
        rem = N2 - U2 - U
        if rem < 0:
            break

        # If even the "best case" (V=W, w=-W) fails d<=n, larger W won't work.
        Wmax0 = isqrt_local(rem // 2)
        if Wmax0 > U:
            Wmax0 = U

        # Full-range test depends on T = 2n - 2U^2 - 2U.
        T = N2 - 2 * U2 - 2 * U

        for sgn in (1, -1):
            startW = 0 if sgn == 1 else 1  # avoid double-counting W=0
            if startW > Wmax0:
                continue

            # Wfull: largest W where V can run all the way up to U (no discriminant needed).
            if T < 0:
                Wfull = -1
            else:
                rt = isqrt_local(1 + 4 * T)
                if sgn == 1:
                    Wfull = (rt - 1) // 2  # W^2 + W <= T
                else:
                    Wfull = (rt + 1) // 2  # W^2 - W <= T
                if Wfull > Wmax0:
                    Wfull = Wmax0

            # --- Full segment: W in [startW..Wfull], V in [W..U] ---
            if Wfull >= startW:
                A = startW
                B = Wfull
                num = B - A + 1

                sumW = p1_local[B] - (p1_local[A - 1] if A else 0)
                sumW2 = p2_local[B] - (p2_local[A - 1] if A else 0)
                sumW3 = p3_local[B] - (p3_local[A - 1] if A else 0)

                sumCnt = num * (U + 1) - sumW
                sumCntW2 = (U + 1) * sumW2 - sumW3

                # sum_{W=A..B} s2[W-1] = sum_{t=A-1..B-1} s2[t]
                if B == 0:
                    sumPrefix = 0
                else:
                    lo = A - 1
                    hi = B - 1
                    if lo < 0:
                        lo = 0
                    sumPrefix = ps2_local[hi] - (ps2_local[lo - 1] if lo > 0 else 0)

                sumSumV2 = num * s2_local[U] - sumPrefix

                contrib = 2 * U2 * sumCnt + 2 * sumCntW2 + 2 * sumSumV2
                add(contrib if mod is None else (contrib % mod))

            # --- Tail segment: compute Vmax via discriminant (per W) ---
            tail_start = Wfull + 1
            if tail_start < startW:
                tail_start = startW
            if tail_start > Wmax0:
                continue

            # D = 8n+1 - 4(U^2+W^2+U+sgn*W).  Using N2=2n:
            # D = 4*N2 + 1 - 4(...)
            base_D = 4 * N2 + 1 - 4 * (U2 + U)

            for W in range(tail_start, Wmax0 + 1):
                D = base_D - 4 * (W * W + sgn * W)
                if D < 0:
                    break

                Vmax = (isqrt_local(D) - 1) // 2
                if Vmax > U:
                    Vmax = U
                if Vmax < W:
                    continue

                cnt = Vmax - W + 1
                sumV2 = s2_local[Vmax] - (s2_local[W - 1] if W else 0)

                W2 = W * W
                contrib = cnt * 2 * (U2 + W2) + 2 * sumV2
                add(contrib if mod is None else (contrib % mod))

    if mod is None:
        return total
    return total % mod


def main() -> None:
    # Test values given in the problem statement:
    assert solve(5) == 48
    assert solve(10**3) == 37048340

    MOD = 433494437
    n = 10**8
    print(solve(n, mod=MOD) % MOD)


if __name__ == "__main__":
    main()
