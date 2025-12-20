#!/usr/bin/env python3

MOD = 10**9 + 7
XOR_SIZE = 64  # since k-1 <= 61 when w=64, fits in 6 bits


def fwt_xor(arr, inverse=False):
    """
    Fast Walsh-Hadamard Transform for XOR convolution.
    If inverse=True, performs the inverse transform.
    """
    n = len(arr)
    step = 1
    while step < n:
        jump = step * 2
        for i in range(0, n, jump):
            for j in range(i, i + step):
                x = arr[j]
                y = arr[j + step]
                arr[j] = (x + y) % MOD
                arr[j + step] = (x - y) % MOD
        step = jump

    if inverse:
        inv_n = pow(n, MOD - 2, MOD)
        for i in range(n):
            arr[i] = arr[i] * inv_n % MOD


def poly_mul(pa, pb):
    """
    Multiply two shifted polynomials:
      pa = (coeff_list, offset)
      pb = (coeff_list, offset)
    where degree of coeff[i] is i - offset.

    Returns (result_coeffs, result_offset).
    """
    a, oa = pa
    b, ob = pb
    if len(a) < len(b):
        a, b = b, a
        oa, ob = ob, oa

    res = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if ai:
            for j, bj in enumerate(b):
                if bj:
                    res[i + j] = (res[i + j] + ai * bj) % MOD
    return res, oa + ob


def poly_pow(poly, exp):
    """
    Raise a shifted polynomial to exp using exponentiation by squaring.
    """
    result = ([1], 0)
    base = poly
    while exp > 0:
        if exp & 1:
            result = poly_mul(result, base)
        exp >>= 1
        if exp:
            base = poly_mul(base, base)
    return result


def compute_R(m, w):
    """
    Compute R(m, w) modulo MOD.
    Each staircase (a,b,k) corresponds to game:
        (b-a) + nimber(k-1)
    Winning condition for Right (moving first):
      - sum(b-a) > 0, OR
      - sum(b-a) == 0 and XOR(nimbers) != 0
    """

    dmax = w - 2  # max possible |b-a|
    diff_count = 2 * dmax + 1
    counts = [[0] * XOR_SIZE for _ in range(diff_count)]

    # Count staircases by (d=b-a, g=k-1)
    for k in range(1, w - 1):
        limit = w - k  # a+b <= limit
        if limit < 2:
            continue
        g = k - 1
        tmax = limit - 2  # max abs(d)
        for t in range(tmax + 1):
            c = (limit - t) // 2
            if c <= 0:
                continue
            counts[dmax + t][g] = (counts[dmax + t][g] + c) % MOD
            if t != 0:
                counts[dmax - t][g] = (counts[dmax - t][g] + c) % MOD

    # FWT over XOR dimension for each diff coefficient
    transformed = []
    for vec in counts:
        v = vec[:]
        fwt_xor(v, inverse=False)
        transformed.append(v)

    # Build 64 polynomials, one per transformed XOR index
    polys = []
    for t in range(XOR_SIZE):
        coeffs = [transformed[d][t] for d in range(diff_count)]
        polys.append((coeffs, dmax))

    # Raise each polynomial to the m-th power
    pow_polys = []
    for t in range(XOR_SIZE):
        pow_polys.append(poly_pow(polys[t], m))

    final_offset = m * dmax
    final_len = 2 * final_offset + 1

    # Align into matrix [t][degree]
    Qhat = [[0] * final_len for _ in range(XOR_SIZE)]
    for t in range(XOR_SIZE):
        coeffs, off = pow_polys[t]
        # should match offset, pad if needed
        if off != final_offset:
            raise ValueError("Unexpected polynomial offset mismatch")
        if len(coeffs) < final_len:
            coeffs += [0] * (final_len - len(coeffs))
        Qhat[t] = coeffs

    # Inverse FWT at each coefficient position, accumulate winning cases
    ans = 0
    for idx in range(final_len):
        vec = [Qhat[t][idx] for t in range(XOR_SIZE)]
        fwt_xor(vec, inverse=True)

        total_diff = idx - final_offset
        if total_diff > 0:
            ans = (ans + sum(vec)) % MOD
        elif total_diff == 0:
            ans = (ans + sum(vec[1:])) % MOD

    return ans


def main():
    # Given examples
    assert compute_R(2, 4) == 7
    assert compute_R(3, 9) == 314104

    # Required output
    print(compute_R(8, 64) % MOD)


if __name__ == "__main__":
    main()
