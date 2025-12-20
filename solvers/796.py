# Project Euler 796: A Grand Shuffle
#
# We compute an expected stopping time for drawing cards without replacement until
# every suit (4), rank (13), and deck design (10) has appeared at least once.
#
# No external libraries are used.

from decimal import Decimal, getcontext, ROUND_HALF_UP
import math


# High precision to make rounding to 8 decimals safe.
getcontext().prec = 80

_sum_ratio_cache = {}


def _sum_choose_ratios(M: int, N: int) -> Decimal:
    """
    Return S(M,N) = sum_{k=0}^{N-1} C(M,k) / C(N,k).

    With a fixed N, this lets us convert:
        E[T] = sum_{k>=0} P(T > k)
    into a small set of sums indexed by M.

    Computed via the recurrence:
        r_0 = 1
        r_k = r_{k-1} * (M-k+1)/(N-k+1)
    where r_k = C(M,k)/C(N,k).
    """
    key = (M, N)
    if key in _sum_ratio_cache:
        return _sum_ratio_cache[key]

    r = Decimal(1)
    s = Decimal(1)  # k=0

    max_k = min(N - 1, M)
    for k in range(1, max_k + 1):
        r *= Decimal(M - k + 1) / Decimal(N - k + 1)
        s += r

    _sum_ratio_cache[key] = s
    return s


def _expectation_from_coeffs(N: int, coeff_by_M: dict[int, int]) -> Decimal:
    """
    Given coefficients such that:
        P(T > k) = sum_M coeff[M] * C(M,k)/C(N,k)
    compute:
        E[T] = sum_{k=0}^{N-1} P(T > k)
             = sum_M coeff[M] * sum_{k=0}^{N-1} C(M,k)/C(N,k)
    """
    total = Decimal(0)
    for M, coef in coeff_by_M.items():
        if coef:
            total += Decimal(coef) * _sum_choose_ratios(M, N)
    return total


def expected_cards_for_all_ranks_single_deck() -> Decimal:
    """
    Sanity check value from the statement:
    One 54-card deck (52 + 2 jokers), draw without replacement until all 13 ranks appear.

    For b missing ranks, allowed cards are:
        4*(13-b)  non-jokers + 2 jokers  = 54 - 4b
    Inclusion-exclusion over ranks gives P(T>k), then tail-sum gives E[T].
    """
    N = 54
    coeff_by_M: dict[int, int] = {}

    for b in range(1, 13 + 1):
        coef = ((-1) ** (b + 1)) * math.comb(13, b)
        M = 54 - 4 * b
        coeff_by_M[M] = coeff_by_M.get(M, 0) + coef

    return _expectation_from_coeffs(N, coeff_by_M)


def expected_cards_for_all_suits_ranks_and_decks() -> Decimal:
    """
    Main target:
    10 decks, each 54 cards (52 + 2 jokers), shuffled together => N = 540 cards.

    Requirements:
    - All 4 suits seen (only possible via the 520 non-jokers)
    - All 13 ranks seen (only possible via the 520 non-jokers)
    - All 10 deck designs seen (possible via non-jokers or jokers)

    For a missing suits, b missing ranks, c missing deck designs:
      allowed non-jokers: (4-a)*(13-b)*(10-c)
      allowed jokers:     2*(10-c)
      total allowed M(a,b,c) = (4-a)*(13-b)*(10-c) + 2*(10-c)

    By symmetry, only (a,b,c) matters, and inclusion-exclusion reduces to summing over
    these counts rather than over all specific subsets.
    """
    N = 540
    coeff_by_M: dict[int, int] = {}

    for a in range(0, 4 + 1):
        ca = math.comb(4, a)
        for b in range(0, 13 + 1):
            cb = math.comb(13, b)
            for c in range(0, 10 + 1):
                if a == 0 and b == 0 and c == 0:
                    continue
                cc = math.comb(10, c)

                # (-1)^(a+b+c+1)
                sign = -1 if ((a + b + c) % 2 == 0) else 1
                coef = sign * ca * cb * cc

                decks_left = 10 - c
                M = (4 - a) * (13 - b) * decks_left + 2 * decks_left
                coeff_by_M[M] = coeff_by_M.get(M, 0) + coef

    return _expectation_from_coeffs(N, coeff_by_M)


def main() -> None:
    # Assert the check value stated in the problem (rounded to 8 decimals).
    check = expected_cards_for_all_ranks_single_deck().quantize(
        Decimal("0.00000001"), rounding=ROUND_HALF_UP
    )
    assert check == Decimal("29.05361725")

    ans = expected_cards_for_all_suits_ranks_and_decks().quantize(
        Decimal("0.00000001"), rounding=ROUND_HALF_UP
    )
    print(ans)


if __name__ == "__main__":
    main()
