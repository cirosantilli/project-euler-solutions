#!/usr/bin/env python3
"""Project Euler 898: Claire Voyant

We compute the optimal probability that Claire guesses the coin toss.

Key idea:
- The optimal decision is the likelihood-ratio test: guess Heads iff
  P(reports|Heads) >= P(reports|Tails).
- With equal priors, the optimal success probability is
      1/2 * sum_reports max(P(r|H), P(r|T)).

For the concrete instance (lie probabilities 25%..75%), probabilities come in
complementary pairs p and (1-p). Pairing them reduces each pair of students to
a 3-outcome independent variable, enabling an efficient meet-in-the-middle.

No external libraries are used.
"""

from __future__ import annotations

from array import array
from bisect import bisect_left
from typing import List, Tuple


Transition = Tuple[
    int, int, float, float
]  # (mul_num, mul_den, prob_under_H, prob_under_T)


def _build_variables_from_percentages(ps: List[int]) -> List[List[Transition]]:
    """Convert lie probabilities (integer percents) into independent variables.

    We pair probabilities p and 100-p when both are present. Each such pair
    becomes a 3-outcome variable corresponding to the log-likelihood contribution
    {-2w, 0, +2w}.

    Any unpaired probability (rare for this problem) becomes a 2-outcome variable.

    Students with p=50% provide no information and are ignored.
    """
    counts = [0] * 101
    for k in ps:
        if not (0 <= k <= 100):
            raise ValueError("percentages must be in [0, 100]")
        counts[k] += 1

    variables: List[List[Transition]] = []

    # Pair k with 100-k
    for k in range(0, 50):
        if k == 0:
            # p=0 would imply a perfect truth-teller; allowed theoretically, but not present here.
            pass
        m = min(counts[k], counts[100 - k])
        if m <= 0:
            continue
        counts[k] -= m
        counts[100 - k] -= m

        a = 100 - k
        b = k
        # Likelihood ratio factors are ((1-p)/p)^2 and its inverse.
        # We keep them as exact integer fractions.
        num_plus = a * a
        den_plus = b * b
        num_minus = b * b
        den_minus = a * a

        # Under true Heads:
        #   + corresponds to (truthful student with p=k%) says H AND
        #                (complement student with p=100-k%) says T.
        # With pairing, these probabilities become (1-p)^2, 2p(1-p), p^2.
        p_plus_H = (a * a) / 10000.0
        p_zero = (2 * a * b) / 10000.0
        p_minus_H = (b * b) / 10000.0

        # Under true Tails, + and - swap.
        p_plus_T = p_minus_H
        p_minus_T = p_plus_H

        for _ in range(m):
            variables.append(
                [
                    (num_minus, den_minus, p_minus_H, p_minus_T),
                    (1, 1, p_zero, p_zero),
                    (num_plus, den_plus, p_plus_H, p_plus_T),
                ]
            )

    # Ignore p=50% students (no information)
    counts[50] = 0

    # Any leftovers become 2-outcome variables.
    for k in range(0, 101):
        for _ in range(counts[k]):
            if k == 0 or k == 100:
                # Degenerate (always truthful or always lying). Not needed here.
                raise ValueError("degenerate p=0 or p=100 not supported")
            a = 100 - k
            b = k
            # report=H factor is (1-p)/p = a/b; report=T factor is inverse.
            num_H, den_H = a, b
            num_T, den_T = b, a
            p_H_under_H = a / 100.0
            p_T_under_H = b / 100.0
            p_H_under_T = b / 100.0
            p_T_under_T = a / 100.0
            variables.append(
                [
                    (num_T, den_T, p_T_under_H, p_T_under_T),
                    (num_H, den_H, p_H_under_H, p_H_under_T),
                ]
            )

    return variables


def _enumerate_half(
    variables: List[List[Transition]],
) -> Tuple[List[int], array, array, int]:
    """Enumerate all combined states for one half.

    Returns:
      keys: list of scaled fixed-point values floor(LR * 2^SHIFT)
      pH:   array of probabilities under true Heads
      pT:   array of probabilities under true Tails
      bound: upper bound on numerator/denominator factors for this half (for SHIFT selection)
    """
    # Bound for denominators/numerators: product of max factor per variable.
    bound = 1
    for var in variables:
        mx = 1
        for mul_n, mul_d, _, _ in var:
            if mul_n > mx:
                mx = mul_n
            if mul_d > mx:
                mx = mul_d
        bound *= mx

    nums: List[int] = [1]
    dens: List[int] = [1]
    pH = array("d", [1.0])
    pT = array("d", [1.0])

    for var in variables:
        new_nums: List[int] = []
        new_dens: List[int] = []
        new_pH = array("d")
        new_pT = array("d")

        # Localize for speed
        append_num = new_nums.append
        append_den = new_dens.append
        append_pH = new_pH.append
        append_pT = new_pT.append

        for i in range(len(nums)):
            n0 = nums[i]
            d0 = dens[i]
            ph0 = pH[i]
            pt0 = pT[i]
            for mul_n, mul_d, ph_mul, pt_mul in var:
                append_num(n0 * mul_n)
                append_den(d0 * mul_d)
                append_pH(ph0 * ph_mul)
                append_pT(pt0 * pt_mul)

        nums = new_nums
        dens = new_dens
        pH = new_pH
        pT = new_pT

    # SHIFT will be chosen later; for now return raw num/den in key computation step.
    # But to keep memory modest, we compute keys after SHIFT is known.
    return nums, dens, pH, pT, bound


def _fixed_point_keys(nums: List[int], dens: List[int], shift: int) -> List[int]:
    """Compute floor((num/den) * 2^shift) for each fraction."""
    out = [0] * len(nums)
    for i, (n, d) in enumerate(zip(nums, dens)):
        out[i] = (n << shift) // d
    return out


def _sort_by_keys(
    keys: List[int], pH: array, pT: array
) -> Tuple[List[int], array, array]:
    """Sort (keys, pH, pT) by keys, returning new arrays."""
    idx = list(range(len(keys)))
    idx.sort(key=keys.__getitem__)

    keys_s = [0] * len(keys)
    pH_s = array("d")
    pT_s = array("d")
    pH_s.extend([0.0] * len(keys))
    pT_s.extend([0.0] * len(keys))

    for j, i in enumerate(idx):
        keys_s[j] = keys[i]
        pH_s[j] = pH[i]
        pT_s[j] = pT[i]

    return keys_s, pH_s, pT_s


def _suffix_sums(arr: array) -> array:
    """Return suffix sums: suf[i] = sum_{j>=i} arr[j]."""
    n = len(arr)
    suf = array("d", [0.0])
    suf.extend([0.0] * n)
    # suf has length n+1; suf[n]=0
    s = 0.0
    for i in range(n - 1, -1, -1):
        s += arr[i]
        suf[i] = s
    return suf


def _kahan_add(total: float, c: float, x: float) -> Tuple[float, float]:
    y = x - c
    t = total + y
    c = (t - total) - y
    return t, c


def optimal_success_probability(ps: List[int]) -> float:
    """Compute Claire's optimal probability of guessing correctly."""
    variables = _build_variables_from_percentages(ps)
    if not variables:
        return 0.5

    mid = len(variables) // 2
    left_vars = variables[:mid]
    right_vars = variables[mid:]

    # Enumerate halves
    l_nums, l_dens, l_pH, l_pT, l_bound = _enumerate_half(left_vars)
    r_nums, r_dens, r_pH, r_pT, r_bound = _enumerate_half(right_vars)

    bound = max(l_bound, r_bound)
    # Choose SHIFT so that 2^SHIFT > bound^2, ensuring the fixed-point mapping is injective
    # over all possible fractions formed within this bound.
    shift = 2 * bound.bit_length() + 4

    # Compute and sort right-half keys + suffix sums
    r_keys = _fixed_point_keys(r_nums, r_dens, shift)

    # Free raw right nums/dens early
    r_nums = []
    r_dens = []

    r_keys, r_pH, r_pT = _sort_by_keys(r_keys, r_pH, r_pT)
    r_suf_H = _suffix_sums(r_pH)
    r_suf_T = _suffix_sums(r_pT)

    # Left-half: compute contributions on the fly
    pHA = 0.0
    cH = 0.0
    pTA = 0.0
    cT = 0.0

    # For each left state with LR = n/d, we need right LR >= d/n.
    for n, d, ph, pt in zip(l_nums, l_dens, l_pH, l_pT):
        thr_key = (d << shift) // n
        j = bisect_left(r_keys, thr_key)
        # Multiply by probability mass in right half that meets threshold.
        pHA, cH = _kahan_add(pHA, cH, ph * r_suf_H[j])
        pTA, cT = _kahan_add(pTA, cT, pt * r_suf_T[j])

    tv = pHA - pTA
    # Numerical guard
    if tv < -1.0:
        tv = -1.0
    elif tv > 1.0:
        tv = 1.0

    return 0.5 * (1.0 + tv)


def main() -> None:
    # Test from the problem statement.
    test = optimal_success_probability([20, 40, 60, 80])
    assert round(test, 3) == 0.832

    # Target instance: 51 students with lie probabilities 25%, 26%, ..., 75%.
    ps = list(range(25, 76))
    ans = optimal_success_probability(ps)
    print(f"{ans:.10f}")


if __name__ == "__main__":
    main()
