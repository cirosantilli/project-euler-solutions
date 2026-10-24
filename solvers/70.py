#!/usr/bin/env python


def compute_phi_linear(limit: int) -> list[int]:
    """
    Linear sieve computing phi[0..limit] in O(limit).

    @formalsw: Returns
    """
    phi = [0] * (limit + 1)
    if limit < 1:
        return phi
    is_comp = [False] * (limit + 1)
    primes: list[int] = []

    phi[1] = 1
    for i in range(2, limit + 1):
        if not is_comp[i]:
            primes.append(i)
            phi[i] = i - 1
        for p in primes:
            ip = i * p
            if ip > limit:
                break
            is_comp[ip] = 1
            if i % p == 0:
                phi[ip] = phi[i] * p
                break
            phi[ip] = phi[i] * (p - 1)
    return phi


def digit_signature(x: int) -> list[int]:
    """
    Count each decimal digit without restricting the number of repetitions.
    Nonpositive inputs have zero counts.

    @formalsw: Returns
    """
    counts = [0] * 10
    while x > 0:
        counts[x % 10] += 1
        x //= 10
    return counts


def solve(limit: int = 9_999_999) -> int:
    """
    Return the least totient-permutation ratio minimizer up to the inclusive bound.
    Return zero if there is no candidate.
    """
    if limit < 2:
        return 0
    phi = compute_phi_linear(limit)

    best_n = 0
    best_phi = 1

    # Digit-length boundaries for n, to quickly reject when phi(n) has different length.
    low = 1
    high = 10

    for n in range(2, limit + 1):
        if n == high:
            low = high
            high *= 10

        pn = phi[n]

        # Permutations must have same number of digits
        if pn < low or pn >= high:
            continue

        # Permutations preserve digit-sum => same mod 9
        if (n - pn) % 9 != 0:
            continue

        if digit_signature(n) != digit_signature(pn):
            continue

        if best_n == 0 or n * best_phi < best_n * pn:
            best_n = n
            best_phi = pn

    return best_n


if __name__ == "__main__":
    sample_phi = compute_phi_linear(87109)
    assert sample_phi[1] == 1
    assert sample_phi[9] == 6
    assert sample_phi[87109] == 79180
    assert digit_signature(87109) == digit_signature(79180)

    print(solve())
