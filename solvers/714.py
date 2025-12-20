#!/usr/bin/env python3
"""Project Euler 714 — Duodigits

We call a natural number a duodigit if its decimal representation uses no more than
2 different digits.

For each n, let d(n) be the smallest positive multiple of n that is a duodigit.
Let D(k) = sum_{n=1..k} d(n).

Task: compute D(50000) and print it in scientific notation rounded to
13 significant digits (12 after the decimal point).

Constraints for this submission:
- No third‑party libraries (standard library only)
- Include asserts for all check values given in the problem statement.
"""

from collections import deque
import math


# ---------------------------
# Helpers
# ---------------------------


def is_duodigit(x: int) -> bool:
    """Return True if x uses at most two distinct decimal digits."""
    s = str(x)
    a = s[0]
    b = None
    for ch in s:
        if ch != a:
            if b is None:
                b = ch
            elif ch != b:
                return False
    return True


def digit_mask_str(s: str) -> int:
    """10-bit mask of digits appearing in s."""
    m = 0
    for ch in s:
        m |= 1 << (ord(ch) - 48)
    return m


# ---------------------------
# Candidate generation
# ---------------------------


def generate_duodigits_upto(max_len: int) -> list[int]:
    """Generate all duodigits with 1..max_len digits (no leading zeros), sorted."""
    res: list[int] = []

    for L in range(1, max_len + 1):
        # repdigits 1..9
        for d in range(1, 10):
            res.append(int(str(d) * L))

        # two-digit duodigits: choose a<b, then enumerate all non-trivial patterns
        for a in range(0, 9):
            for b in range(a + 1, 10):
                if a == 0:
                    if L == 1:
                        continue
                    # leading digit cannot be 0 => MSB must be b
                    topbit = 1 << (L - 1)
                    limit = 1 << (L - 1)
                    for tail in range(limit):
                        if tail == limit - 1:
                            continue  # all b => repdigit
                        mask = topbit | tail
                        num = 0
                        for pos in range(L - 1, -1, -1):
                            num = num * 10 + (b if (mask >> pos) & 1 else 0)
                        res.append(num)
                else:
                    limit = 1 << L
                    for mask in range(1, limit - 1):  # exclude all-a and all-b
                        num = 0
                        for pos in range(L - 1, -1, -1):
                            num = num * 10 + (b if (mask >> pos) & 1 else a)
                        res.append(num)

    res.sort()
    return res


# ---------------------------
# Scientific notation formatting
# ---------------------------


def format_sci_13(x: int) -> str:
    """Scientific notation with 13 significant digits (12 after decimal), rounded."""
    s = str(x)
    exp = len(s) - 1
    sig = 13

    if len(s) <= sig:
        cut = s.ljust(sig, "0")
    else:
        cut = s[:sig]
        next_digit = ord(s[sig]) - 48
        if next_digit >= 5:
            cut_list = list(cut)
            i = sig - 1
            while i >= 0:
                if cut_list[i] != "9":
                    cut_list[i] = chr(ord(cut_list[i]) + 1)
                    break
                cut_list[i] = "0"
                i -= 1
            if i < 0:
                exp += 1
                cut = "1" + ("0" * (sig - 1))
            else:
                cut = "".join(cut_list)

    mantissa = cut[0] + "." + cut[1:]
    return f"{mantissa}e{exp}"


# ---------------------------
# {0,1} BFS (used for n divisible by 10)
# ---------------------------


def smallest_multiple_01(k: int) -> str:
    """Smallest positive multiple of k using only digits {0,1}, leading digit 1."""
    if k == 1:
        return "1"

    prev = [-1] * k
    prev_digit = [-1] * k

    start = 1 % k
    q = deque([start])
    prev[start] = -2
    prev_digit[start] = 1

    while q:
        r = q.popleft()
        if r == 0:
            break

        r10 = (r * 10) % k

        nr = r10
        if prev[nr] == -1:
            prev[nr] = r
            prev_digit[nr] = 0
            q.append(nr)

        nr = (r10 + 1) % k
        if prev[nr] == -1:
            prev[nr] = r
            prev_digit[nr] = 1
            q.append(nr)

    # reconstruct
    r = 0
    out = []
    while True:
        out.append(str(prev_digit[r]))
        pr = prev[r]
        if pr == -2:
            break
        r = pr
    out.reverse()
    return "".join(out)


def best_multiple_0d(m: int, cache_01: list[str]) -> str:
    """For fixed m, find smallest multiple that uses digits {0,d} for some d=1..9.

    Any {0,d}-digit number equals d * X where X is a {0,1}-digit number (no carries).
    So we find minimal X divisible by m/gcd(m,d), then map 1->d.
    """
    best = None
    for d in range(1, 10):
        g = math.gcd(m, d)
        k = m // g
        pattern = cache_01[k]
        cand = "".join("0" if ch == "0" else str(d) for ch in pattern)
        if best is None or (len(cand), cand) < (len(best), best):
            best = cand
    return best


# ---------------------------
# Correctness fallback (rare)
# ---------------------------


def bfs_duodigit_multiple(n: int) -> int:
    """Guaranteed-correct smallest duodigit multiple of n.

    A state is (mask, rem) where mask is the set of digits used so far
    (bitmask over 0..9) and rem is the current remainder mod n.

    We only allow masks that contain <= 2 digits.
    BFS (by length) + digit expansion in ascending order yields the smallest
    value among shortest lengths, hence the smallest integer overall.

    This fallback is expected to trigger extremely rarely for n <= 50000.
    """

    # Precompute which digit masks are valid (1 or 2 bits set).
    good_mask = [False] * 1024
    for m in range(1, 1024):
        if (m & (m - 1)) == 0:
            good_mask[m] = True
        else:
            mm = m & (m - 1)
            if mm and (mm & (mm - 1)) == 0:
                good_mask[m] = True

    def key(mask: int, rem: int) -> int:
        return mask * n + rem

    q = deque()
    parent: dict[int, int] = {}
    digit_of: dict[int, int] = {}

    # Start with 1..9 (no leading zero)
    for d in range(1, 10):
        mask = 1 << d
        rem = d % n
        k0 = key(mask, rem)
        if k0 not in parent:
            parent[k0] = -1
            digit_of[k0] = d
            q.append((mask, rem))
        if rem == 0:
            return d

    while q:
        mask, rem = q.popleft()
        for dig in range(0, 10):
            nm = mask | (1 << dig)
            if not good_mask[nm]:
                continue
            nr = (rem * 10 + dig) % n
            k1 = key(nm, nr)
            if k1 in parent:
                continue
            parent[k1] = key(mask, rem)
            digit_of[k1] = dig
            if nr == 0:
                out = []
                cur = k1
                while cur != -1:
                    out.append(str(digit_of[cur]))
                    cur = parent[cur]
                out.reverse()
                return int("".join(out))
            q.append((nm, nr))

    raise RuntimeError("Unreachable: every n has a duodigit multiple")


# ---------------------------
# Main computation
# ---------------------------


def compute_D(limit: int) -> int:
    # Empirically sufficient for all n not divisible by 10 (as commonly used for this problem).
    candidates = generate_duodigits_upto(15)

    # Precompute smallest {0,1} multiples for k<=5000 (needed because n<=50000 => n/10<=5000)
    cache_01 = [""] * 5001
    for k in range(1, 5001):
        cache_01[k] = smallest_multiple_01(k)

    dvals = [0] * (limit + 1)

    cand = candidates  # local alias
    for n in range(1, limit + 1):
        if n % 10 == 0:
            m = n // 10
            base = best_multiple_0d(m, cache_01)
            dvals[n] = int(base + "0")
            continue

        if is_duodigit(n):
            dvals[n] = n
            continue

        # brute scan over sorted duodigits
        found = None
        for x in cand:
            if x % n == 0:
                found = x
                break

        # The 15-digit list is expected to succeed for n % 10 != 0 in this range.
        # Keep a guaranteed-correct fallback for robustness.
        if found is None:
            found = bfs_duodigit_multiple(n)

        dvals[n] = found

    # Asserts for all example/check values given in the statement
    assert dvals[12] == 12
    assert dvals[102] == 1122
    assert dvals[103] == 515
    assert dvals[290] == 11011010
    assert dvals[317] == 211122

    # Prefix sums D(k)
    pref = [0] * (limit + 1)
    s = 0
    for i in range(1, limit + 1):
        s += dvals[i]
        pref[i] = s

    assert pref[110] == 11047
    assert pref[150] == 53312
    assert pref[500] == 29570988

    return pref[limit]


def main() -> None:
    total = compute_D(50000)
    print(format_sci_13(total))


if __name__ == "__main__":
    main()
