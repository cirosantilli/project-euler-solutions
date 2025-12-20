#!/usr/bin/env python3
"""
Project Euler 744: What? Where? When?

Key idea:
- The order in which envelopes are opened is a uniformly random permutation.
- Let Q be the number of questions needed for either side to reach n points
  in an i.i.d. Bernoulli(p) sequence (ignoring the RED card).
- The RED card position K is uniform on {1,2,...,2n+1} and independent of answers.
  The game ends normally iff K > Q, hence:

    f(n,p) = P(K > Q) = E[(2n+1 - Q)/(2n+1)] = 1 - E[Q]/(2n+1).

We compute E[Q] exactly for moderate n using a closed-form sum over the losing score,
evaluated safely in log-space; for very large n with noticeable bias we use that the
underdog winning is astronomically unlikely, so E[Q] is essentially the negative
binomial mean n / max(p,1-p).
"""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_UP
from typing import Union

Number = Union[float, Decimal]


EXACT_N_LIMIT = 200_000  # O(n) log-space sum is fine up to this range


def _expected_questions_exact(n: int, p: float) -> float:
    """Exact E[Q] in O(n) using log-sum-exp over terminal distributions."""
    if not (0.0 < p < 1.0):
        return float(n)  # deterministic: one side scores every time
    q = 1.0 - p
    logp = math.log(p)
    logq = math.log(q)

    # Terminal distributions:
    # viewers win with k successes before the nth failure:
    #   A_k = C(n+k-1, k) * p^k * q^n
    # expert  wins with k failures  before the nth success:
    #   B_k = C(n+k-1, k) * q^k * p^n
    #
    # E[Q] = sum_{k=0}^{n-1} (n+k) * (A_k + B_k)
    logs_a = [0.0] * n
    logs_b = [0.0] * n
    m = -1e300

    lgamma_n = math.lgamma(n)
    for k in range(n):
        # log C(n+k-1, k) = lgamma(n+k) - lgamma(n) - lgamma(k+1)
        logc = math.lgamma(n + k) - lgamma_n - math.lgamma(k + 1)
        la = logc + k * logp + n * logq
        lb = logc + k * logq + n * logp
        logs_a[k] = la
        logs_b[k] = lb
        if la > m:
            m = la
        if lb > m:
            m = lb

    s = 0.0
    for k in range(n):
        s += (n + k) * (math.exp(logs_a[k] - m) + math.exp(logs_b[k] - m))
    return math.exp(m) * s


def _expected_questions_approx(n: int, p: Decimal) -> Decimal:
    """
    Fast approximation for huge n:
    if max(p,1-p) = a, then typically the a-side reaches n first and
    Q is essentially a negative binomial waiting time with mean n/a.
    """
    if p <= 0:
        return Decimal(n)
    if p >= 1:
        return Decimal(n)
    q = Decimal(1) - p
    a = p if p >= q else q
    return Decimal(n) / a


def f(n: int, p: Number) -> Number:
    """Compute f(n,p) as 1 - E[Q]/(2n+1), exact for moderate n, approximate for huge n."""
    denom = 2 * n + 1
    if n <= EXACT_N_LIMIT:
        pf = float(p)
        eq = _expected_questions_exact(n, pf)
        return 1.0 - eq / float(denom)
    else:
        pd = p if isinstance(p, Decimal) else Decimal(str(p))
        eq = _expected_questions_approx(n, pd)
        return Decimal(1) - eq / Decimal(denom)


def _round10_str(x: Number) -> str:
    """Round to 10 digits after the decimal point, returning a fixed-format string."""
    if isinstance(x, Decimal):
        q = x.quantize(Decimal("0.0000000001"), rounding=ROUND_HALF_UP)
        # quantize keeps trailing zeros with format 'f'
        return format(q, "f")
    else:
        return f"{x:.10f}"


def _self_test() -> None:
    # Test values given in the problem statement (rounded to 10 decimal places)
    assert _round10_str(f(6, 0.5)) == "0.2851562500"
    assert _round10_str(f(10, 3.0 / 7.0)) == "0.2330040743"
    assert _round10_str(f(10_000, 0.3)) == "0.2857499982"


def main() -> None:
    _self_test()
    n = 10**11
    p = Decimal("0.4999")
    ans = f(n, p)
    print(_round10_str(ans))


if __name__ == "__main__":
    main()
