#!/usr/bin/env python3
"""
Project Euler 946 - Continued Fraction Fraction

We need the sum of the first 10**8 coefficients of the continued fraction of

    beta = (2*alpha + 3) / (3*alpha + 2)

where alpha has continued fraction digits starting

    alpha = [2; 1,1,2, 1,1,1,2, 1,1,1,1,1,2, ...]

and the number of 1s between consecutive 2s are consecutive prime numbers.

No external libraries are used.
"""

import math
from collections import deque


# --------- Homographic continued-fraction spigot utilities ---------


def floor_div(n: int, d: int) -> int:
    """Floor division for integers with possibly negative numerator/denominator (d != 0)."""
    if d < 0:
        n, d = -n, -d
    if n >= 0:
        return n // d
    # For negative n, Python's // is already floor, but keep explicit and safe:
    return -((-n + d - 1) // d)


def normalize(A: int, B: int, C: int, D: int):
    """
    A small normalization that keeps the pair (C, C+D) from being simultaneously negative.
    This is enough for the safe extraction condition used here.
    """
    if C < 0 and (C + D) < 0:
        return -A, -B, -C, -D
    return A, B, C, D


def step_transition(state, inp: int):
    """
    Consume one continued-fraction digit 'inp' of alpha and then emit as many continued-fraction
    digits of beta as possible, returning the new stable state plus the emitted digits.

    State represents the homography:
        y = (A*x + B) / (C*x + D)
    where x is the remaining continued-fraction tail of alpha, and we know x >= 1.
    """
    A, B, C, D = state
    A, B, C, D = normalize(A, B, C, D)

    # Consume one input CF digit: multiply by M(inp) = [[inp, 1], [1, 0]]
    A, B, C, D = A * inp + B, A, C * inp + D, C

    outs = []
    while True:
        A, B, C, D = normalize(A, B, C, D)

        # With x in [1, +inf), endpoints are at x=1 and x=+inf.
        denom_inf = C
        denom_1 = C + D
        if denom_inf != 0 and denom_1 != 0 and ((denom_inf > 0) == (denom_1 > 0)):
            q_inf = floor_div(A, denom_inf)  # floor at x = +inf
            q_1 = floor_div(A + B, denom_1)  # floor at x = 1
            if q_inf == q_1:
                q = q_inf
                outs.append(q)
                # Extract digit q: y = q + 1/y', so new homography is:
                # y' = 1/(y-q) => matrix update:
                A, B, C, D = C, D, A - q * C, B - q * D
                continue

        break

    return (A, B, C, D), tuple(outs)


def stable_initial():
    """
    Initial homography for beta = (2*alpha+3)/(3*alpha+2).
    """
    state = (2, 3, 3, 2)
    # In this problem, no digits are extractable before consuming input,
    # but keep the same convention as step_transition (stable between inputs).
    return state


# --------- Alpha digit source (2, then prime-run ones, then 2, repeating) ---------


class PrimeGen:
    __slots__ = ("primes", "cand")

    def __init__(self):
        self.primes = []
        self.cand = 2

    def next(self) -> int:
        if not self.primes:
            self.primes = [2]
            self.cand = 3
            return 2
        n = self.cand
        while True:
            r = int(math.isqrt(n))
            is_p = True
            for p in self.primes:
                if p > r:
                    break
                if n % p == 0:
                    is_p = False
                    break
            if is_p:
                self.primes.append(n)
                self.cand = n + 2
                return n
            n += 2
            self.cand = n


class AlphaCursor:
    """
    Cursor over alpha's continued fraction digits:
        2,
        then for each consecutive prime p: (1 repeated p times), then 2.

    stage:
        0 => next digit is the initial 2
        1 => next digits are 1s (ones_left > 0)
        2 => next digit is separator 2 (after the 1-run)
    """

    __slots__ = ("prime_gen", "stage", "ones_left")

    def __init__(self):
        self.prime_gen = PrimeGen()
        self.stage = 0
        self.ones_left = 0

    def can_skip_ones(self) -> bool:
        return self.stage == 1 and self.ones_left > 0

    def skip_ones(self, k: int) -> None:
        # assumes stage==1 and 0 < k <= ones_left
        self.ones_left -= k
        if self.ones_left == 0:
            self.stage = 2

    def next_digit(self) -> int:
        if self.stage == 0:
            # initial 2
            self.stage = 1
            self.ones_left = self.prime_gen.next()
            return 2
        if self.stage == 1:
            self.ones_left -= 1
            if self.ones_left == 0:
                self.stage = 2
            return 1
        # stage == 2: separator 2, then start next prime-run of ones
        self.stage = 1
        self.ones_left = self.prime_gen.next()
        return 2


# --------- Build finite-state transducer + binary lifting for runs of 1s ---------


def prepare_tables():
    init = stable_initial()

    # Collect reachable stable states under inputs {1,2}
    states = [init]
    idx = {init: 0}
    q = deque([init])

    while q:
        st = q.popleft()
        for inp in (1, 2):
            nst, _ = step_transition(st, inp)
            if nst not in idx:
                idx[nst] = len(states)
                states.append(nst)
                q.append(nst)

    S = len(states)

    # Base transitions for input 1 and 2
    next1 = [0] * S
    cnt1 = [0] * S
    sum1 = [0] * S

    next2 = [0] * S
    cnt2 = [0] * S
    sum2 = [0] * S

    for sid, st in enumerate(states):
        nst, outs = step_transition(st, 1)
        next1[sid] = idx[nst]
        cnt1[sid] = len(outs)
        sum1[sid] = sum(outs)

        nst, outs = step_transition(st, 2)
        next2[sid] = idx[nst]
        cnt2[sid] = len(outs)
        sum2[sid] = sum(outs)

    # Binary lifting for repeated 1s: apply 2^k ones at once
    # Choose enough bits (primes needed here are far below 2^25)
    MAX_POW = 25
    pow_next = [next1]
    pow_cnt = [cnt1]
    pow_sum = [sum1]

    for _ in range(1, MAX_POW):
        prev_n = pow_next[-1]
        prev_c = pow_cnt[-1]
        prev_s = pow_sum[-1]
        cur_n = [0] * S
        cur_c = [0] * S
        cur_s = [0] * S
        for sid in range(S):
            mid = prev_n[sid]
            cur_n[sid] = prev_n[mid]
            cur_c[sid] = prev_c[sid] + prev_c[mid]
            cur_s[sid] = prev_s[sid] + prev_s[mid]
        pow_next.append(cur_n)
        pow_cnt.append(cur_c)
        pow_sum.append(cur_s)

    return states, idx[init], pow_next, pow_cnt, pow_sum, next2, cnt2, sum2


def consume_ones_with_limit(
    state_id: int, max_ones: int, out_limit: int, pow_next, pow_cnt, pow_sum
):
    """
    Consume up to max_ones ones, but do not emit more than out_limit beta-digits.
    Because emitted-count is monotone in consumed input digits, we can greedily
    take the largest 2^k blocks that fit.
    Returns (consumed_ones, new_state_id, emitted_count, emitted_sum).
    """
    if max_ones <= 0 or out_limit <= 0:
        return 0, state_id, 0, 0

    consumed = 0
    emitted_c = 0
    emitted_s = 0

    bit = max_ones.bit_length() - 1
    while bit >= 0:
        step = 1 << bit
        if consumed + step <= max_ones:
            c = pow_cnt[bit][state_id]
            if c <= out_limit:
                out_limit -= c
                consumed += step
                emitted_c += c
                emitted_s += pow_sum[bit][state_id]
                state_id = pow_next[bit][state_id]
        bit -= 1

    return consumed, state_id, emitted_c, emitted_s


# --------- Public helpers ---------


def first_k_beta_digits(k: int):
    """Compute the first k continued-fraction coefficients of beta (digit-by-digit)."""
    cursor = AlphaCursor()
    state = stable_initial()
    out = []
    while len(out) < k:
        a = cursor.next_digit()
        state, outs = step_transition(state, a)
        if outs:
            out.extend(outs)
    return out[:k]


def sum_first_n_beta_digits(n: int, cutoff: int = 2000) -> int:
    """
    Sum the first n coefficients of beta's continued fraction.

    Uses binary lifting to skip large prime-length runs of 1s until we're close,
    then finishes digit-by-digit to handle the rare case where a single consumed
    input digit emits 2 output digits.
    """
    states, init_id, pow_next, pow_cnt, pow_sum, next2, cnt2, sum2 = prepare_tables()
    cursor = AlphaCursor()
    state_id = init_id

    total_sum = 0
    total_cnt = 0
    safe_target = max(0, n - cutoff)

    # Fast path: avoid stepping through each '1' in long runs.
    while total_cnt < safe_target:
        if cursor.can_skip_ones():
            remaining_allowed = safe_target - total_cnt
            max_ones = cursor.ones_left
            used, state_id, c, s = consume_ones_with_limit(
                state_id, max_ones, remaining_allowed, pow_next, pow_cnt, pow_sum
            )
            if used == 0:
                break
            cursor.skip_ones(used)
            total_cnt += c
            total_sum += s
            if used < max_ones:
                break  # next 1 would exceed safe_target in output count
        else:
            # Next alpha digit is a '2' (initial or separator)
            c = cnt2[state_id]
            if total_cnt + c > safe_target:
                break
            total_sum += sum2[state_id]
            total_cnt += c
            state_id = next2[state_id]
            _ = cursor.next_digit()  # consume that 2

    # Finish carefully, digit-by-digit.
    state = states[state_id]
    while total_cnt < n:
        a = cursor.next_digit()
        state, outs = step_transition(state, a)
        for q in outs:
            if total_cnt >= n:
                break
            total_sum += q
            total_cnt += 1

    return total_sum


def _self_test():
    # Problem statement test:
    expected = [0, 1, 5, 6, 16, 9, 1, 10, 16, 11]
    got = first_k_beta_digits(10)
    assert got == expected, (got, expected)
    assert sum(got) == 75


def main():
    _self_test()
    print(sum_first_n_beta_digits(10**8))


if __name__ == "__main__":
    main()
