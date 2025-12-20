#!/usr/bin/env python3
"""Project Euler 841 - Regular Star Polygons

Compute
    sum_{n=3..34} A(F_{n+1}, F_{n-1})
rounded to 10 digits after the decimal point.

No external libraries are used.

---

Let s = pi/p.
A direct derivation (from sector integration and parity along rays) gives:

    A(p,q) = p * ( tan(q*s) + 2*(-1)^q * sum_{k=1..q-1} (-1)^k * tan(k*s) )

For the large Fibonacci inputs, evaluating that expression naively is
numerically ill-conditioned: the bracket is ~1/p and is formed by cancellation
among O(q) terms of size O(1).

We eliminate the cancellation using the exact identity

    tan(x) - tan(x - s) = sin(s) / (cos(x) * cos(x - s))

and the telescoping representation

    tan(m*s) = sum_{k=1..m} (tan(k*s) - tan((k-1)*s)).

After algebra (grouping terms by these increments), the bracket becomes an
alternating sum of those *increments*, which are all O(s). Pulling out sin(s)
leads to a well-conditioned form:

    A(p,q) = (p*sin(s)) * sum_{k=1..q} (-1)^{k+q} / (cos(k*s) * cos((k-1)*s)).

Now p*sin(s) is ~pi (order 1), and the remaining sum is also order 1.

Implementation details:
- compute cos(k*s) directly with math.cos for best accuracy
- use Kahan summation for both the inner sum and the final total
"""

from __future__ import annotations

import math


def A(p: int, q: int) -> float:
    """Return A(p,q) for coprime integers p>2q>0."""
    if not (isinstance(p, int) and isinstance(q, int) and p > 2 * q and q > 0):
        raise ValueError("Need integers with p > 2q > 0")
    if math.gcd(p, q) != 1:
        raise ValueError("Need coprime p and q")

    s = math.pi / p
    p_sin_s = p * math.sin(s)

    # sign for odd k in (-1)^{k+q}
    sign_odd = 1.0 if (q & 1) else -1.0

    cos_prev = 1.0  # cos(0*s)

    # Kahan summation for C
    C = 0.0
    c = 0.0

    for k in range(1, q + 1):
        cos_k = math.cos(k * s)
        sign = sign_odd if (k & 1) else -sign_odd
        term = sign / (cos_k * cos_prev)

        y = term - c
        t = C + y
        c = (t - C) - y
        C = t

        cos_prev = cos_k

    return p_sin_s * C


def fibonacci_upto(n: int) -> list[int]:
    """Return [F_0, F_1, ..., F_n] with F_0=0, F_1=1."""
    if n < 0:
        return []
    F = [0, 1]
    for _ in range(2, n + 1):
        F.append(F[-1] + F[-2])
    return F


def main() -> None:
    # Test values given in the problem statement
    assert f"{A(8, 3):.4f}" == "9.9411"
    assert f"{A(130021, 50008):.4f}" == "10.9210"

    F = fibonacci_upto(35)  # need up to F_35

    total = 0.0
    c = 0.0
    for n in range(3, 35):
        p = F[n + 1]
        q = F[n - 1]
        a = A(p, q)

        # Kahan summation for the outer total
        y = a - c
        t = total + y
        c = (t - total) - y
        total = t

    print(f"{total:.10f}")


if __name__ == "__main__":
    main()
