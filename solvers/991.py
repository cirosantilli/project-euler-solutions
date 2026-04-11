from __future__ import annotations

from math import gcd, isqrt, sqrt

LIMIT = 10_000_000


# The problem's equation is the asymmetric one:
#   a/(b+c) + b/(c+a) + c/(a+c) = 4
# so the last two terms share the same denominator.
#
# Writing t = b + c gives
#   a/t + t/(a+c) = 4,
# hence
#   t^2 - 4(a+c)t + a(a+c) = 0.
# Therefore the discriminant
#   (a+c)(3a+4c)
# must be a perfect square.
#
# Let g = gcd(a+c, c), and write
#   a+c = g u,
#   c   = g v,
# with gcd(u, v) = 1. Then
#   (a+c)(3a+4c) = g^2 * u * (3u+v),
# and gcd(u, 3u+v) = gcd(u, v) = 1.
# So both factors must be squares:
#   u = m^2,
#   3u + v = n^2,
# giving
#   c = g(n^2 - 3m^2),
#   a = g(4m^2 - n^2),
#   b = g(5m^2 - n^2 ± mn).
#
# Positivity requires
#   3m^2 < n^2 < 4m^2,
# and for the minus branch additionally
#   5m^2 - n^2 - mn > 0.
#
# Every solution is a positive multiple of a unique primitive solution (g = 1).


def primitive_solutions(limit: int) -> list[int]:
    """Return the sums of all primitive positive solutions with sum <= limit."""
    sums: list[int] = []

    # Plus branch:
    #   s = a+b+c = 6m^2 - n^2 + mn.
    # Since n < 2m, we have s > 4m^2, so m <= sqrt(limit/4).
    m_max = isqrt(limit // 4) + 2
    for m in range(1, m_max + 1):
        n_min = isqrt(3 * m * m) + 1
        n_max = 2 * m - 1
        for n in range(n_min, n_max + 1):
            if gcd(m, n) != 1:
                continue

            a = 4 * m * m - n * n
            c = n * n - 3 * m * m
            b = 5 * m * m - n * n + m * n
            s = a + b + c

            if a <= 0 or b <= 0 or c <= 0:
                continue
            if s <= limit:
                sums.append(s)

    # Minus branch.
    # Let k = 2m - n, so n = 2m - k and gcd(m, n) = gcd(m, k).
    # Then
    #   s = a+b+c = k(5m-k),
    #   c > 0  <=>  m > (2+sqrt(3)) k,
    #   b > 0  <=>  m < ((5+sqrt(21))/2) k.
    # This narrows the search to O(sqrt(limit)).
    alpha = 2.0 + sqrt(3.0)
    beta = (5.0 + sqrt(21.0)) / 2.0

    k = 1
    while True:
        low = int(alpha * k) + 1
        while (2 * low - k) ** 2 <= 3 * low * low:
            low += 1

        high_pos = int(beta * k)
        while high_pos > 0 and not (
            -high_pos * high_pos + 5 * high_pos * k - k * k > 0
        ):
            high_pos -= 1

        high_sum = (limit + k * k) // (5 * k)
        high = min(high_pos, high_sum)

        if low > high_sum:
            break

        for m in range(low, high + 1):
            if gcd(m, k) != 1:
                continue

            n = 2 * m - k
            a = 4 * m * m - n * n
            c = n * n - 3 * m * m
            b = 5 * m * m - n * n - m * n
            s = a + b + c

            if a <= 0 or b <= 0 or c <= 0:
                continue
            if s <= limit:
                sums.append(s)

        k += 1

    return sums


def solve(limit: int = LIMIT) -> int:
    primitive = primitive_solutions(limit)
    total = 0
    for s in primitive:
        count = limit // s
        total += s * count * (count + 1) // 2
    return total


# -----------------------------
# Self-checks
# -----------------------------


def brute_force(limit: int) -> int:
    total = 0
    for a in range(1, limit + 1):
        for b in range(1, limit + 1 - a):
            max_c = limit - a - b
            for c in range(1, max_c + 1):
                lhs_num = a * (a + c) + (b + c) * (b + c)
                lhs_den = (b + c) * (a + c)
                if lhs_num == 4 * lhs_den:
                    total += a + b + c
    return total


def run_tests() -> None:
    # Primitive examples from the parameterization.
    assert (4 * 4 * 4 - 7 * 7, 5 * 4 * 4 - 7 * 7 - 4 * 7, 7 * 7 - 3 * 4 * 4) == (
        15,
        3,
        1,
    )
    assert (4 * 4 * 4 - 7 * 7, 5 * 4 * 4 - 7 * 7 + 4 * 7, 7 * 7 - 3 * 4 * 4) == (
        15,
        59,
        1,
    )

    # Small-range exact cross-checks against brute force.
    assert solve(18) == 0
    assert solve(19) == brute_force(19)
    assert solve(75) == brute_force(75)
    assert solve(200) == brute_force(200)


if __name__ == "__main__":
    run_tests()
    print(solve())
