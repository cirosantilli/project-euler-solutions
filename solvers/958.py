#!/usr/bin/env python3
"""
Project Euler 958: Euclid's Labour

We use the subtraction-only Euclidean algorithm. For coprime inputs (n, m),
the algorithm ends at (1, 1). Reversing one subtraction step turns (a, b)
into either (a+b, a) or (a+b, b) (keeping the larger component first).

So d(n, m) is exactly the length of the unique path in this "addition" tree
from (n, m) down to (1, 1). To minimize d(n, m) for fixed n, we search
backwards from the penultimate state (2, 1) and look for the first time
we can reach a state (n, m). If several m appear at that same minimal depth,
we take the smallest m.

Key pruning:
- With s steps remaining from (x, y), the largest possible first component is
  F_{s+1}*x + F_s*y (F are Fibonacci numbers).
- The first component strictly increases each step, so once x > n we can stop.
  Also, even with the slowest growth (always keeping the smaller second),
  the first component is at least x + s*y, so if that already exceeds n we stop.
- A necessary condition for ever reaching first component n from (x, y) is that
  n can be written as a nonnegative integer combination of x and y:
      n = a*x + b*y  with a, b >= 0.
  We test this quickly using the modular inverse of x mod y.
  Instead of recomputing inverses from scratch, we carry Bézout coefficients
  (p, q) such that p*x + q*y = 1, which update in O(1) along the tree; then
  p is an inverse of x modulo y, and a = (n*p) mod y gives the minimal a >= 0.

No external libraries are used.
"""

from __future__ import annotations


def f(n: int) -> int:
    """
    Return f(n): the smallest positive m, gcd(n,m)=1, minimizing the number of
    subtraction steps in the Euclidean algorithm.

    The optimal m is always < n (if m >= n, reducing modulo n lowers the step count),
    so we only need to consider states with first component == n and second < n.
    """
    if n <= 1:
        raise ValueError("n must be >= 2")

    # Precompute Fibonacci numbers up to a safe limit.
    # We only need around 70 for n ~ 1e12, but make it generous.
    fib = [0, 1]
    for _ in range(200):
        fib.append(fib[-1] + fib[-2])

    # Depth is counted as steps from (2,1) to (x,y). Total subtraction steps is depth+1.
    # The maximum first component after depth d from (2,1) is Fib[d+3].
    depth = 0
    while fib[depth + 3] < n:
        depth += 1

    INF = 10**30
    n_local = n
    fib_local = fib

    while True:
        best_m = INF

        # Stack entries: (x, y, p, q, rem)
        # Invariant: x > y >= 1, gcd(x,y)=1, and p*x + q*y = 1.
        stack = [(2, 1, 0, 1, depth)]

        stack_append = stack.append
        stack_pop = stack.pop

        while stack:
            x, y, p, q, rem = stack_pop()

            # Since y never decreases along descendants, this is a safe bound for tie-breaking.
            if y >= best_m:
                continue

            if x == n_local:
                best_m = y
                continue

            if rem == 0:
                continue

            if x > n_local:
                continue

            # Even with the slowest possible growth (always keeping y as the second),
            # x increases by at least y per remaining step.
            if x + rem * y > n_local:
                continue

            # Upper bound: even with the fastest possible growth, can we reach n?
            # max_x = F_{rem+1}*x + F_{rem}*y
            if fib_local[rem + 1] * x + fib_local[rem] * y < n_local:
                continue

            # Nonnegative linear-combination test using the inverse of x mod y.
            # Since p*x + q*y = 1, we have p ≡ x^{-1} (mod y).
            a = (n_local * p) % y
            if a * x > n_local:
                continue

            xp = x + y
            rem1 = rem - 1

            # Children:
            # 1) (x+y, y): keep the smaller second (often helps find smaller m sooner)
            #    Bézout update: p' = p, q' = q - p.
            # 2) (x+y, x): second becomes x
            #    Bézout update: p' = q, q' = p - q.
            # Push in reverse order so child (x+y, y) is explored first (LIFO stack).
            stack_append((xp, x, q, p - q, rem1))
            stack_append((xp, y, p, q - p, rem1))

        if best_m != INF:
            return best_m

        depth += 1


def _run_tests() -> None:
    # Test values from the problem statement.
    assert f(7) == 2
    assert f(89) == 34
    assert f(8191) == 1856


def main() -> None:
    _run_tests()
    n = 10**12 + 39
    print(f(n))


if __name__ == "__main__":
    main()
