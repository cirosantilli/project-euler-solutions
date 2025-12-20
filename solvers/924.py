#!/usr/bin/env python3
"""
Project Euler 924: Larger Digit Permutation II

Let B(n) be the smallest number larger than n that can be formed by rearranging
digits of n, or 0 if no such number exists.

Define a_0 = 0 and a_n = a_{n-1}^2 + 2 for n > 0.
Let U(N) = sum_{n=1..N} B(a_n).

Print U(10^16) modulo 1_000_000_007.

No external libraries are used.
"""

MOD = 1_000_000_007


# -------------------- Next permutation (no leading zeros) --------------------


def _next_permutation_inplace(digs):
    """In-place next lexicographic permutation of list of digits; return True if advanced."""
    i = len(digs) - 2
    while i >= 0 and digs[i] >= digs[i + 1]:
        i -= 1
    if i < 0:
        return False
    j = len(digs) - 1
    while digs[j] <= digs[i]:
        j -= 1
    digs[i], digs[j] = digs[j], digs[i]
    l, r = i + 1, len(digs) - 1
    while l < r:
        digs[l], digs[r] = digs[r], digs[l]
        l += 1
        r -= 1
    return True


def B(n):
    """Problem-defined B(n)."""
    digs = [ord(c) - 48 for c in str(n)]
    if not _next_permutation_inplace(digs):
        return 0
    y = 0
    for d in digs:
        y = y * 10 + d
    return y


# -------------------- Fixed-width next permutation (leading zeros allowed) --------------------


def next_perm_fixed_int(x, k, buf):
    """
    Treat x as exactly k decimal digits (leading zeros allowed),
    write digits into buf (length k), apply next permutation,
    and return the permuted integer. Return None if no next permutation exists.
    """
    t = x
    for i in range(k - 1, -1, -1):
        buf[i] = t % 10
        t //= 10

    i = k - 2
    while i >= 0 and buf[i] >= buf[i + 1]:
        i -= 1
    if i < 0:
        return None

    j = k - 1
    while buf[j] <= buf[i]:
        j -= 1

    buf[i], buf[j] = buf[j], buf[i]
    l, r = i + 1, k - 1
    while l < r:
        buf[l], buf[r] = buf[r], buf[l]
        l += 1
        r -= 1

    y = 0
    for d in buf:
        y = y * 10 + d
    return y


# -------------------- Direct small-N (for the provided test) --------------------


def U_direct_mod(N):
    """Compute U(N) mod MOD directly using big integers (only for small N)."""
    a = 0
    s = 0
    for _ in range(N):
        a = a * a + 2
        s = (s + B(a)) % MOD
    return s


# -------------------- Sum of a_n mod MOD via cycle detection --------------------


def sum_a_mod(N):
    """
    Compute sum_{n=1..N} (a_n mod MOD) mod MOD for:
      a_0 = 0
      a_n = a_{n-1}^2 + 2 (mod MOD)

    Uses first-repeat detection with a dictionary (cycle is small).
    """
    x = 0
    seen = {0: 0}
    states = [0]  # states[i] = a_i mod MOD

    while True:
        nxt = (x * x + 2) % MOD
        idx = len(states)
        if nxt in seen:
            mu = seen[nxt]
            lam = idx - seen[nxt]
            break
        seen[nxt] = idx
        states.append(nxt)
        x = nxt

    # pref[i] = sum_{t=1..i} a_t (mod MOD), with pref[0] = 0
    pref = [0] * len(states)
    for i in range(1, len(states)):
        pref[i] = (pref[i - 1] + states[i]) % MOD

    if N < len(states):
        return pref[N]

    # Sum of a_1..a_{mu-1}
    base_before = pref[mu - 1] if mu > 0 else 0
    # Sum of one full cycle a_mu..a_{mu+lam-1}
    cycle_sum = (pref[mu + lam - 1] - base_before) % MOD

    cycle_terms = N - mu + 1
    full = cycle_terms // lam
    rem = cycle_terms % lam

    total = (base_before + (full % MOD) * cycle_sum) % MOD
    if rem:
        total = (total + (pref[mu + rem - 1] - base_before)) % MOD
    return total


# -------------------- Delta decomposition for B(a_n) --------------------


def delta_small(N):
    """
    Exact contribution of delta_n = B(a_n) - a_n (mod MOD) for small n,
    where fixed-width (10-digit) handling would introduce leading-zero artifacts.
    """
    a = 0
    s = 0
    for n in range(1, min(N, 5) + 1):
        a = a * a + 2
        s = (s + (B(a) - a)) % MOD
    return s


def delta10_and_bad(N):
    """
    For n >= 6, a_n has >= 10 digits. If the last 10 digits are not fully
    non-increasing, the pivot for next-permutation lies within those 10 digits,
    so:
        B(a_n) - a_n = next_perm(last10) - last10.

    There is exactly one residue in the 10-digit cycle where last10 has no next
    permutation; those indices form an arithmetic progression. This function
    returns:
      (sum_delta10_over_n>=6, first_bad_index, step)
    where step is the cycle length modulo 10^10.
    """
    if N <= 5:
        return 0, None, None

    k = 10
    m = 10**k
    step = 8 * (5 ** (k - 2))  # cycle length (lambda) modulo 10^k

    # start at n=6: compute a_6 mod 10^10
    x = 0
    for _ in range(6):
        x = (x * x + 2) % m
    start = x

    total_terms = N - 5  # n=6..N inclusive
    q, r = divmod(total_terms, step)

    buf = [0] * k
    cycle_sum = 0
    rem_sum = 0
    bad_step = None  # 1-based within the step-length cycle, relative to n=6

    # Scan exactly one full cycle (step states), starting at a_6
    for i in range(1, step + 1):
        y = next_perm_fixed_int(x, k, buf)
        if y is None:
            if bad_step is not None:
                raise AssertionError("More than one bad position in the 10-digit cycle")
            bad_step = i
            # contributes 0 to delta10 (handled separately with 11 digits)
        else:
            d = y - x
            cycle_sum = (cycle_sum + d) % MOD
            if i <= r:
                rem_sum = (rem_sum + d) % MOD

        x = (x * x + 2) % m

    # Must return to the starting residue after one cycle
    assert x == start
    assert bad_step is not None

    first_bad_n = bad_step + 5  # since i=1 corresponds to n=6
    total = ((q % MOD) * cycle_sum + rem_sum) % MOD
    return total, first_bad_n, step


def delta_bad_11(N, first_bad_n, step):
    """
    Handle exactly those indices where last10 digits are non-increasing (no 10-digit next perm).
    At those indices, the next permutation pivot is the 11th digit from the end, so
        delta = next_perm(last11) - last11.

    The bad indices are:
        n = first_bad_n + t*step,  t>=0

    Modulo 10^11, the main cycle length is 5*step, so the subsequence sampled every
    'step' has period 5. We compute the 5 deltas once and then count them up to N.
    """
    if first_bad_n is None or N < first_bad_n:
        return 0

    m11 = 10**11
    targets = [first_bad_n + t * step for t in range(5)]
    max_n = targets[-1]

    # Simulate a_n mod 10^11 up to max_n and record the 5 needed values
    x = 0
    vals = [0] * 5
    idx = 0
    for n in range(1, max_n + 1):
        x = (x * x + 2) % m11
        if n == targets[idx]:
            vals[idx] = x
            idx += 1
            if idx == 5:
                break
    assert idx == 5

    buf = [0] * 11
    deltas = [0] * 5
    for i, v in enumerate(vals):
        y = next_perm_fixed_int(v, 11, buf)
        # If this were None, pivot would be deeper than 11; we never see that.
        assert y is not None
        deltas[i] = (y - v) % MOD

    # Count how many bad indices <= N
    T = 1 + (N - first_bad_n) // step
    base = T // 5
    rem = T % 5

    total = 0
    for i, d in enumerate(deltas):
        c = base + (1 if i < rem else 0)
        total = (total + (c % MOD) * d) % MOD
    return total


def solve(N):
    # U(N) = sum a_n + sum (B(a_n)-a_n)  (mod MOD)
    s_a = sum_a_mod(N)

    d_small = delta_small(N)
    d10, first_bad_n, step = delta10_and_bad(N)
    d_bad = delta_bad_11(N, first_bad_n, step)

    return (s_a + d_small + d10 + d_bad) % MOD


def main():
    # Asserts for the statement's explicit test values
    assert B(245) == 254
    assert B(542) == 0
    assert U_direct_mod(10) == 543870437

    print(solve(10**16))


if __name__ == "__main__":
    main()
