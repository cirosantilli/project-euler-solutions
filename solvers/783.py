#!/usr/bin/env python3
"""
Project Euler 783 - Urns

We start with k*n white balls. For t=1..n:
  - add k black balls
  - remove 2k uniformly random balls (without replacement)

Let B_t be the number of black balls removed on turn t.
We need E(n,k) = E[ sum_{t=1..n} B_t^2 ] and output E(10^6, 10) rounded
to the nearest whole number.

No external libraries are used.
"""


def expected_sum_square(n: int, k: int) -> float:
    """
    Compute E(n,k) = E[ sum_{t=1..n} B_t^2 ].

    Technique:
      Track first and second moments of the number of black balls in the urn
      at the *start* of each turn (before adding k black balls).
      Conditional on the current black count, B_t is hypergeometric, so its
      second moment is a quadratic function of that black count. This lets us
      update moments in O(1) per turn.
    """
    if n <= 0 or k <= 0:
        raise ValueError("n and k must be positive integers")

    # m = number of draws (balls removed) each turn
    m = float(2 * k)

    # Moments of X_t = #black at start of turn t (before adding k black balls)
    mu = 0.0  # E[X_t]
    s2 = 0.0  # E[X_t^2]

    # Population size after adding k black balls on turn t:
    # M_t = k * (n - t + 2). This starts at k*(n+1) and decreases by k each turn.
    M = float(k * (n + 1))

    # Kahan summation for better numerical stability of the final sum
    total = 0.0
    comp = 0.0

    kf = float(k)

    for _ in range(n):
        # After adding k black balls: Y = X + k
        Ey = mu + kf
        Ey2 = s2 + 2.0 * kf * mu + kf * kf

        # Hypergeometric parameters:
        # population size M, success count Y, draws m
        # E[B^2 | Y] = c1*Y + c2*Y^2, where
        # c1 = m(M-m) / (M(M-1)), c2 = m(m-1) / (M(M-1))
        denom = M * (M - 1.0)
        c1 = m * (M - m) / denom
        c2 = m * (m - 1.0) / denom

        Eb2 = c1 * Ey + c2 * Ey2

        # Kahan add Eb2 into total
        y = Eb2 - comp
        t = total + y
        comp = (t - total) - y
        total = t

        # Moment update for next turn
        # X_next = Y - B
        alpha = (M - m) / M  # 1 - m/M

        mu_next = alpha * Ey

        # E[X_next^2] = E[Y^2] - 2*(m/M)*E[Y^2] + E[B^2]
        #            = (1 - 2m/M + c2)*E[Y^2] + c1*E[Y]
        coeff_y2 = 1.0 - 2.0 * m / M + c2
        s2_next = coeff_y2 * Ey2 + c1 * Ey

        mu, s2 = mu_next, s2_next
        M -= kf

    return total


def solve() -> None:
    # Test value from the statement
    assert abs(expected_sum_square(2, 2) - 9.6) < 1e-12

    # Required computation
    value = expected_sum_square(10**6, 10)

    # "Round to nearest whole number" (half-up for positive values)
    print(int(value + 0.5))


if __name__ == "__main__":
    solve()
