from math import comb

MOD = 1234567891


def _trim(poly):
    """Remove trailing zero coefficients, leaving at least one term."""
    while len(poly) > 1 and poly[-1] == 0:
        poly.pop()
    return poly


def _add_to(dst, src, mod=None):
    """Add polynomial src into dst in place."""
    if len(dst) < len(src):
        dst.extend([0] * (len(src) - len(dst)))
    if mod is None:
        for i, value in enumerate(src):
            dst[i] += value
    else:
        for i, value in enumerate(src):
            dst[i] = (dst[i] + value) % mod


def _mul_one_minus_q(poly, mod=None):
    """Return poly * (1 - q)."""
    out = [0] * (len(poly) + 1)
    for i, value in enumerate(poly):
        out[i] += value
        out[i + 1] -= value
    if mod is not None:
        out = [value % mod for value in out]
    return _trim(out)


def _mul(a, b, max_degree, mod=None):
    """Multiply polynomials, discarding terms above max_degree."""
    if not a or not b:
        return [0]
    out = [0] * (min(len(a) + len(b) - 2, max_degree) + 1)
    for i, ai in enumerate(a):
        if ai == 0:
            continue
        last_j = min(len(b) - 1, max_degree - i)
        for j in range(last_j + 1):
            bj = b[j]
            if bj:
                out[i + j] += ai * bj
                if mod is not None:
                    out[i + j] %= mod
    if mod is not None:
        out = [value % mod for value in out]
    return _trim(out)


def _block_count(length, cost):
    """
    Number of positive blocks of this length and cost.

    A block with entries x_i has sum 2*cost and every x_i <= cost.
    Inclusion-exclusion is simple because at most one entry can exceed cost.
    """
    if cost <= 0 or 2 * cost < length:
        return 0
    total = comb(2 * cost - 1, length - 1)
    too_large = 0 if cost < length else comb(cost - 1, length - 1)
    return total - length * too_large


def _block_numerator(length, mod=None):
    """
    Numerator P_length(q) for the block generating function

        sum_t A(length, t) q^t = P_length(q) / (1 - q)^length.
    """
    coeffs = [0] * (length + 1)
    for j in range(length + 1):
        value = 0
        for i in range(j + 1):
            sign = -1 if i % 2 else 1
            value += sign * comb(length, i) * _block_count(length, j - i)
        coeffs[j] = value % mod if mod is not None else value
    return _trim(coeffs)


def _numerator_for_all_valid_vectors(n, mod=None):
    """
    Return N_n(q), where the generating function for all valid vectors is

        H_n(q) = N_n(q) / (1 - q)^n.

    The coefficient of q^t in H_n is the number of valid vectors whose total
    number of overtakes is exactly 2*t.
    """
    block_num = [None] * (n + 1)
    for length in range(2, n + 1):
        block_num[length] = _block_numerator(length, mod)

    total = [[] for _ in range(n + 1)]      # valid prefixes of this length
    zero_end = [[] for _ in range(n + 1)]   # valid prefixes that may start a block
    total[0] = [1]
    zero_end[0] = [1]

    for pos in range(n + 1):
        if pos < n and total[pos]:
            add_zero = _mul_one_minus_q(total[pos], mod)
            _add_to(total[pos + 1], add_zero, mod)
            _add_to(zero_end[pos + 1], add_zero, mod)

        if zero_end[pos]:
            for length in range(2, n - pos + 1):
                product = _mul(zero_end[pos], block_num[length], pos + length, mod)
                _add_to(total[pos + length], product, mod)

    return total[n]


def count_tuples(n, k, mod=None):
    """Compute F(n, k), optionally modulo mod."""
    max_cost = k // 2
    numerator = _numerator_for_all_valid_vectors(n, mod)

    answer = 0
    for degree, coeff in enumerate(numerator):
        if coeff == 0 or degree > max_cost:
            continue
        ways_up_to_cost = comb(max_cost - degree + n, n)
        if mod is None:
            answer += coeff * ways_up_to_cost
        else:
            answer = (answer + coeff * (ways_up_to_cost % mod)) % mod
    return answer


def _run_tests():
    assert count_tuples(3, 4) == 8
    assert count_tuples(12, 34) == 2457178250


def main():
    _run_tests()
    print(count_tuples(123, 4567891, MOD))


if __name__ == "__main__":
    main()
