#!/usr/bin/env python
from functools import lru_cache

MOD = 10**9 + 7
MAX_N = 50
MAX_TERMS = (MAX_N + 1) // 2
MAX_CARRY = 25
CARRY_MIN = -MAX_CARRY
CARRY_MAX = MAX_CARRY


def build_binom(limit: int) -> list[list[int]]:
    """Build binomial coefficients C[n][k] for 0 <= n <= limit."""
    comb = [[0] * (limit + 1) for _ in range(limit + 1)]
    for n in range(limit + 1):
        comb[n][0] = comb[n][n] = 1
        for k in range(1, n):
            comb[n][k] = (comb[n - 1][k - 1] + comb[n - 1][k]) % MOD
    return comb


def convolve_small(poly: list[int], width: int) -> list[int]:
    """Convolve a polynomial with (1 + x + ... + x^(width-1))."""
    out = [0] * (len(poly) + width - 1)
    for i, value in enumerate(poly):
        if value == 0:
            continue
        for digit in range(width):
            out[i + digit] = (out[i + digit] + value) % MOD
    return out


def build_sum_tables(limit: int) -> list[list[list[int]]]:
    """
    F[p][q][m] = number of ways to write m as
      - a sum of p variables in [0, 9], and
      - a sum of q variables in [0, 8].
    """
    ways_0_to_9 = [None] * (limit + 1)
    ways_0_to_9[0] = [1]
    for p in range(1, limit + 1):
        ways_0_to_9[p] = convolve_small(ways_0_to_9[p - 1], 10)

    tables = [[None] * (limit + 1) for _ in range(limit + 1)]
    for p in range(limit + 1):
        tables[p][0] = ways_0_to_9[p]
        for q in range(1, limit + 1):
            tables[p][q] = convolve_small(tables[p][q - 1], 9)
    return tables


BINOM = build_binom(MAX_TERMS)
SUM_TABLES = build_sum_tables(2 * MAX_TERMS)


@lru_cache(maxsize=None)
def transitions(
    active_left: int, active_right: int, carry: int
) -> tuple[tuple[int, int, int, int], ...]:
    """
    Return all transitions for one decimal column.

    State meaning:
      - active_left / active_right: how many numbers are still alive at the
        current digit position on each side.
      - carry: signed difference carry coming from lower positions.

    A transition chooses how many numbers continue to the next column and how
    the current digits are assigned, producing a new signed carry.
    """
    if active_left == 0 and active_right == 0:
        return ()

    result: list[tuple[int, int, int, int]] = []

    for next_left in range(active_left + 1):
        choose_left = BINOM[active_left][next_left]
        ending_left = active_left - next_left

        for next_right in range(active_right + 1):
            choose_terms = (choose_left * BINOM[active_right][next_right]) % MOD
            continuing = next_left + next_right
            ending = (active_left - next_left) + (active_right - next_right)
            counts = SUM_TABLES[continuing][ending]

            # If the column difference is delta = sum_left - sum_right, then
            # the next signed carry is carry' = (carry + delta) / 10.
            # After the standard digit-shifts described in the README, the
            # needed sum index becomes:
            #   index = delta - ending_left + 9 * active_right
            #         = 10 * carry' - carry - ending_left + 9 * active_right.
            base = -carry - ending_left + 9 * active_right

            for next_carry in range(CARRY_MIN, CARRY_MAX + 1):
                index = 10 * next_carry + base
                if 0 <= index < len(counts):
                    ways = counts[index]
                    if ways:
                        weight = (choose_terms * ways) % MOD
                        result.append((next_left, next_right, next_carry, weight))

    return tuple(result)


def solve(limit: int) -> int:
    """Return A(limit)."""
    # dp[length][(a, b, c)] = number of ways to reach this state using exactly
    # 'length' characters so far, including digits, plus signs, and '='.
    dp: list[dict[tuple[int, int, int], int]] = [dict() for _ in range(limit + 1)]

    # Choose the initial number of terms on each side.
    for left_terms in range(1, MAX_TERMS + 1):
        for right_terms in range(1, MAX_TERMS + 1):
            base_length = left_terms + right_terms - 1  # plus signs + '='
            if base_length <= limit:
                state = (left_terms, right_terms, 0)
                dp[base_length][state] = (dp[base_length].get(state, 0) + 1) % MOD

    answer = 0
    for used_length in range(limit + 1):
        current = dp[used_length]
        if not current:
            continue

        answer = (answer + current.get((0, 0, 0), 0)) % MOD

        for (active_left, active_right, carry), ways_so_far in list(current.items()):
            if ways_so_far == 0 or (active_left == 0 and active_right == 0):
                continue

            next_length = used_length + active_left + active_right
            if next_length > limit:
                continue

            bucket = dp[next_length]
            for next_left, next_right, next_carry, weight in transitions(
                active_left, active_right, carry
            ):
                state = (next_left, next_right, next_carry)
                bucket[state] = (bucket.get(state, 0) + ways_so_far * weight) % MOD

    return answer


def run_self_checks() -> None:
    assert solve(3) == 9
    assert solve(5) == 171
    assert solve(7) == 4878


if __name__ == "__main__":
    run_self_checks()
    print(solve(50))
