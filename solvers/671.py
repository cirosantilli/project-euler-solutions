#!/usr/bin/env python3
"""Project Euler 671: Colouring a Loop.

We tile a 2×n loop with tiles of size 1×1, 1×2, 1×3 (horizontal only) in k colours.
Rules:
  - full cover, no overlap
  - no four corners meeting at a point
  - adjacent tiles must be of different colours

We cut the loop at a boundary between columns and model a boundary state. To count
loop tilings, we sum length-n closed walks that start and end at the same boundary
state. The colour symmetry allows us to handle two cases separately:
  * top/bottom boundary colours equal
  * top/bottom boundary colours different (ordered)
Each case uses fixed distinguished colours (A, or A/B) and treats other colours as
anonymous labels. We build a small transfer matrix for each case and take the
appropriate diagonal entries of T^n.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

MOD = 1_000_004_321

# Row state encoding: 0 = free, 1 = incoming tile ends here, 2 = incoming tile continues.
State = Tuple[int, int, int, int, int]


def _next_from_incoming(row_state: int) -> int:
    if row_state == 1:
        return 0
    if row_state == 2:
        return 1
    raise ValueError("row_state must be incoming")


def _next_from_length(length: int) -> int:
    if length == 1:
        return 0
    if length == 2:
        return 1
    if length == 3:
        return 2
    raise ValueError("length must be 1, 2, or 3")


def _normalize_colors(fixed: int, top: int, bottom: int) -> Tuple[int, int]:
    mapping: Dict[int, int] = {}
    next_label = fixed

    def norm(label: int) -> int:
        nonlocal next_label
        if label < fixed:
            return label
        if label not in mapping:
            mapping[label] = next_label
            next_label += 1
        return mapping[label]

    return norm(top), norm(bottom)


def _valid_state(state: State) -> bool:
    v_prev, t_state, b_state, top_color, bottom_color = state
    if t_state not in (0, 1, 2) or b_state not in (0, 1, 2):
        return False
    if v_prev == 1:
        if t_state != 0 or b_state != 0:
            return False
        if top_color != bottom_color:
            return False
    else:
        if top_color == bottom_color:
            return False
    return True


def _transitions(state: State, k: int, fixed: int) -> Dict[State, int]:
    v_prev, t_state, b_state, top_color, bottom_color = state

    if not _valid_state(state):
        return {}

    in_t = t_state != 0
    in_b = b_state != 0

    total_other = k - fixed
    other_labels = sorted({c for c in (top_color, bottom_color) if c >= fixed})
    m = len(other_labels)
    unused_other = total_other - m

    existing_labels = list(range(fixed)) + other_labels

    out: Dict[State, int] = {}

    def add(
        v_cur: int,
        t_next: int,
        b_next: int,
        new_top: int,
        new_bottom: int,
        weight: int,
    ) -> None:
        if weight <= 0:
            return
        # No-four-corners rule at the boundary between previous and current column.
        if v_prev == 0 and v_cur == 0 and not in_t and not in_b:
            return
        if v_cur == 1 and (t_next != 0 or b_next != 0):
            return
        top_n, bottom_n = _normalize_colors(fixed, new_top, new_bottom)
        ns = (v_cur, t_next, b_next, top_n, bottom_n)
        out[ns] = (out.get(ns, 0) + weight) % MOD

    def single_choices(forbidden: Iterable[int]) -> Iterable[Tuple[int, int, bool]]:
        forbid = set(forbidden)
        for label in existing_labels:
            if label not in forbid:
                yield (label, 1, False)
        if unused_other > 0:
            yield (-1, unused_other, True)

    def pair_choices(forbid_top: Iterable[int], forbid_bottom: Iterable[int]):
        forbid_t = set(forbid_top)
        forbid_b = set(forbid_bottom)
        existing_t = [label for label in existing_labels if label not in forbid_t]
        existing_b = [label for label in existing_labels if label not in forbid_b]

        for ct in existing_t:
            for cb in existing_b:
                if ct != cb:
                    yield (ct, cb, 1)
        if unused_other > 0:
            for ct in existing_t:
                yield (ct, -1, unused_other)
            for cb in existing_b:
                yield (-1, cb, unused_other)
            if unused_other > 1:
                yield (-1, -1, unused_other * (unused_other - 1))

    if in_t and in_b:
        t_next = _next_from_incoming(t_state)
        b_next = _next_from_incoming(b_state)
        add(0, t_next, b_next, top_color, bottom_color, 1)
        return out

    if in_t and not in_b:
        t_next = _next_from_incoming(t_state)
        forbidden = {bottom_color, top_color}
        for label, mult, is_new in single_choices(forbidden):
            bottom_c = fixed + m if is_new else label
            for length in (1, 2, 3):
                b_next = _next_from_length(length)
                add(0, t_next, b_next, top_color, bottom_c, mult)
        return out

    if in_b and not in_t:
        b_next = _next_from_incoming(b_state)
        forbidden = {top_color, bottom_color}
        for label, mult, is_new in single_choices(forbidden):
            top_c = fixed + m if is_new else label
            for length in (1, 2, 3):
                t_next = _next_from_length(length)
                add(0, t_next, b_next, top_c, bottom_color, mult)
        return out

    # Both rows free.
    forbidden = {top_color, bottom_color}
    for label, mult, is_new in single_choices(forbidden):
        c = fixed + m if is_new else label
        add(1, 0, 0, c, c, mult)

    for lt in (1, 2, 3):
        t_next = _next_from_length(lt)
        for lb in (1, 2, 3):
            b_next = _next_from_length(lb)
            for ct, cb, mult in pair_choices({top_color}, {bottom_color}):
                if ct == -1:
                    ct_label = fixed + m
                else:
                    ct_label = ct
                if cb == -1:
                    cb_label = fixed + m + 1 if ct == -1 else fixed + m
                else:
                    cb_label = cb
                add(0, t_next, b_next, ct_label, cb_label, mult)

    return out


def _build_matrix(k: int, fixed: int, start_states: Iterable[State]):
    queue: List[State] = []
    states: List[State] = []
    index: Dict[State, int] = {}

    for s in start_states:
        top, bottom = _normalize_colors(fixed, s[3], s[4])
        ns = (s[0], s[1], s[2], top, bottom)
        if not _valid_state(ns):
            continue
        if ns not in index:
            index[ns] = len(states)
            states.append(ns)
            queue.append(ns)

    qi = 0
    while qi < len(queue):
        s = queue[qi]
        qi += 1
        for ns in _transitions(s, k, fixed):
            if ns not in index:
                index[ns] = len(states)
                states.append(ns)
                queue.append(ns)

    size = len(states)
    mat = [[0] * size for _ in range(size)]
    for s in states:
        i = index[s]
        for ns, w in _transitions(s, k, fixed).items():
            j = index[ns]
            mat[i][j] = (mat[i][j] + w) % MOD

    return mat, index


def _mat_mul(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
    n = len(A)
    out = [[0] * n for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        for k in range(n):
            a = Ai[k]
            if a:
                Bk = B[k]
                for j in range(n):
                    out[i][j] = (out[i][j] + a * Bk[j]) % MOD
    return out


def _mat_pow(A: List[List[int]], exp: int) -> List[List[int]]:
    n = len(A)
    result = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    base = A
    e = exp
    while e > 0:
        if e & 1:
            result = _mat_mul(result, base)
        e //= 2
        if e:
            base = _mat_mul(base, base)
    return result


def count_loop(k: int, n: int) -> int:
    # Case 1: boundary colours equal (fixed A).
    fixed_same = 1
    same_start = (1, 0, 0, 0, 0)
    mat_same, idx_same = _build_matrix(k, fixed_same, [same_start])
    pow_same = _mat_pow(mat_same, n)
    same_count = pow_same[idx_same[same_start]][idx_same[same_start]]

    # Case 2: boundary colours different (fixed A on top, B on bottom).
    fixed_diff = 2
    diff_starts = [(0, t, b, 0, 1) for t in (0, 1, 2) for b in (0, 1, 2)]
    mat_diff, idx_diff = _build_matrix(k, fixed_diff, diff_starts)
    pow_diff = _mat_pow(mat_diff, n)
    diff_count = 0
    for s in diff_starts:
        if s in idx_diff:
            i = idx_diff[s]
            diff_count = (diff_count + pow_diff[i][i]) % MOD

    marked = (k * same_count + k * (k - 1) * diff_count) % MOD
    inv_n = _modinv(n, MOD)
    return (marked * inv_n) % MOD


def _modinv(a: int, mod: int) -> int:
    """Modular inverse via extended gcd; assumes gcd(a, mod) == 1."""
    t0, t1 = 0, 1
    r0, r1 = mod, a % mod
    while r1 != 0:
        q = r0 // r1
        r0, r1 = r1, r0 - q * r1
        t0, t1 = t1, t0 - q * t1
    if r0 != 1:
        raise ValueError("inverse does not exist")
    return t0 % mod


def main() -> None:
    assert count_loop(4, 3) == 104
    assert count_loop(5, 7) == 3_327_300
    assert count_loop(6, 101) == 75_309_980

    n = 10_004_003_002_001
    print(count_loop(10, n))


if __name__ == "__main__":
    main()
