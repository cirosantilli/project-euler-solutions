#!/usr/bin/env python3
"""Project Euler 884: Removing Cubes

Compute S(10^17) where D(n) is the number of steps obtained by repeatedly
subtracting the largest perfect cube not exceeding the current value.

No external libraries are used.
"""

from __future__ import annotations


def icbrt(n: int) -> int:
    """Return floor(cuberoot(n)) for n >= 0."""
    if n <= 0:
        return 0
    # Double precision is accurate enough for our ranges; adjust to be exact.
    x = int(round(n ** (1.0 / 3.0)))
    # Adjust around the floating estimate.
    while (x + 1) * (x + 1) * (x + 1) <= n:
        x += 1
    while x * x * x > n:
        x -= 1
    return x


def d_greedy(n: int) -> int:
    """Directly compute D(n) by simulating the subtraction process (for tests)."""
    steps = 0
    while n > 0:
        k = icbrt(n)
        n -= k * k * k
        steps += 1
    return steps


def build_base_prefix(limit: int) -> list[int]:
    """Build F(n) = sum_{m=0}^{n-1} D(m) for n in [0..limit]."""
    d = [0] * limit
    for i in range(limit):
        d[i] = d_greedy(i)
    pref = [0] * (limit + 1)
    s = 0
    for i in range(limit):
        s += d[i]
        pref[i + 1] = s
    return pref


class PrefixSumD:
    """Computes F(N) = sum_{n=0}^{N-1} D(n) using the cube-interval recurrence.

    Let delta_k = (k+1)^3 - k^3.

    For any N > 1, with K = floor(cuberoot(N-1)) and L = N - K^3:
      F(N) = (N-1) + sum_{k=1}^{K-1} F(delta_k) + F(L)

    We precompute prefix sums of F(delta_k) for small k (only up to the
    maximum cube root that can appear in the recurrence for our target).
    """

    def __init__(
        self, base_limit: int, delta_prefix: list[int], memo_limit: int = 5_000_000
    ):
        self.base_pref = build_base_prefix(base_limit)
        self.delta_prefix = delta_prefix  # delta_prefix[t] = sum_{k=1..t} F(delta_k)
        self.memo_limit = memo_limit
        self._memo: dict[int, int] = {}

    def F(self, N: int) -> int:
        """Return F(N) = sum_{n=0}^{N-1} D(n)."""
        if N <= 0:
            return 0
        if N < len(self.base_pref):
            return self.base_pref[N]
        if N <= self.memo_limit:
            cached = self._memo.get(N)
            if cached is not None:
                return cached

        K = icbrt(N - 1)
        K3 = K * K * K
        L = N - K3

        # K is small for all recursive calls (<= max index of delta_prefix).
        res = (N - 1) + self.delta_prefix[K - 1] + self.F(L)

        if N <= self.memo_limit:
            self._memo[N] = res
        return res


def compute_S(N: int) -> int:
    """Compute S(N) = sum_{1<=n<N} D(n). Since D(0)=0, this equals F(N)."""
    if N <= 1:
        return 0

    K_target = icbrt(N - 1)

    # Largest delta among k <= K_target.
    max_delta = 3 * K_target * K_target + 3 * K_target + 1
    # Maximum cube root that can appear when computing F(delta_k) (k can be huge).
    K0_max = icbrt(max_delta - 1)

    # We only need delta_prefix up to K0_max, because every recursive query uses
    # K <= K0_max (except the final top-level step where we use the full sum).
    delta_prefix = [0] * (K0_max + 1)

    # F helper (uses delta_prefix as it is being built).
    solver = PrefixSumD(base_limit=64, delta_prefix=delta_prefix)

    total_F_delta = 0
    # Build total_F_delta = sum_{k=1..K_target-1} F(delta_k)
    for k in range(1, K_target):
        delta_k = 3 * k * k + 3 * k + 1
        f_delta = solver.F(delta_k)
        total_F_delta += f_delta
        if k <= K0_max:
            delta_prefix[k] = total_F_delta

    # Now compute F(N) using the recurrence once at the top level:
    K3 = K_target * K_target * K_target
    L = N - K3
    return (N - 1) + total_F_delta + solver.F(L)


def _run_tests() -> None:
    # Test values from the problem statement.
    assert d_greedy(100) == 4
    s100 = sum(d_greedy(n) for n in range(1, 100))
    assert s100 == 512


def main() -> None:
    _run_tests()
    N = 10**17
    ans = compute_S(N)
    print(ans)


if __name__ == "__main__":
    main()
