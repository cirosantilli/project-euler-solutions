#!/usr/bin/env python3
"""
Project Euler 790 - Clock Grid

Compute C(10^5) exactly without iterating over the full 50515093 x 50515093 grid.

Key idea:
The final hour at each grid point depends only on how many of the first t rectangles cover it (mod 12).
So we can ignore update order and compute the overlap-count distribution mod 12 across the whole grid.
"""

from array import array
import sys

M = 50515093  # grid side length and RNG modulus
S0 = 290797


def compute_C(t: int) -> int:
    """Return C(t) as defined in the problem statement."""
    if t == 0:
        return 12 * M * M

    # Generate rectangles and store sweep-line events:
    # Each rectangle contributes +1 on [x_lo, x_hi+1) and -1 on [x_hi+1, ...) for the same y-interval.
    s = S0
    x_set = {0, M}
    y_set = {0, M}
    events = []  # (x, shift_mod12, y_lo, y_hi1) with shift 1 for +1, 11 for -1

    for _ in range(t):
        x1 = s
        s = (s * s) % M
        x2 = s
        s = (s * s) % M
        y1 = s
        s = (s * s) % M
        y2 = s
        s = (s * s) % M

        if x1 <= x2:
            xl, xh = x1, x2
        else:
            xl, xh = x2, x1

        if y1 <= y2:
            yl, yh = y1, y2
        else:
            yl, yh = y2, y1

        xh1 = xh + 1
        yh1 = yh + 1

        x_set.add(xl)
        x_set.add(xh1)
        y_set.add(yl)
        y_set.add(yh1)

        events.append((xl, 1, yl, yh1))  # entering: +1
        events.append((xh1, 11, yl, yh1))  # leaving: -1 ≡ +11 (mod 12)

    x_vals = sorted(x_set)
    y_vals = sorted(y_set)
    del x_set, y_set

    # Coordinate-compress y boundaries for segment tree ranges
    y_index = {v: i for i, v in enumerate(y_vals)}
    events_idx = [(x, sh, y_index[yl], y_index[yh1]) for (x, sh, yl, yh1) in events]
    del events, y_index
    events_idx.sort(key=lambda e: e[0])

    # Segment tree over y-intervals [y_vals[i], y_vals[i+1]) (discrete point counts = difference)
    m = len(y_vals) - 1
    size = 4 * m + 5

    # seg[node*12 + r] = total y-length in this node with overlap-count ≡ r (mod 12)
    seg = array("Q", [0]) * (12 * size)
    lazy = bytearray(size)  # pending rotation 0..11

    sys.setrecursionlimit(1_000_000)

    def build(node: int, l: int, r: int) -> None:
        base = node * 12
        if r - l == 1:
            seg[base] = y_vals[l + 1] - y_vals[l]  # initially overlap count = 0
            return
        mid = (l + r) >> 1
        left = node << 1
        build(left, l, mid)
        build(left + 1, mid, r)
        seg[base] = seg[left * 12] + seg[(left + 1) * 12]

    build(1, 0, m)

    buf = [0] * 12
    a = seg
    lz = lazy

    def _apply(node: int, shift: int) -> None:
        """Rotate residue buckets by +shift (mod 12) on this node."""
        if shift == 0:
            return
        shift %= 12
        base = node * 12

        # Fast paths for ±1, which dominate.
        if shift == 1:
            tmp = a[base + 11]
            a[base + 11] = a[base + 10]
            a[base + 10] = a[base + 9]
            a[base + 9] = a[base + 8]
            a[base + 8] = a[base + 7]
            a[base + 7] = a[base + 6]
            a[base + 6] = a[base + 5]
            a[base + 5] = a[base + 4]
            a[base + 4] = a[base + 3]
            a[base + 3] = a[base + 2]
            a[base + 2] = a[base + 1]
            a[base + 1] = a[base]
            a[base] = tmp
        elif shift == 11:
            tmp = a[base]
            a[base] = a[base + 1]
            a[base + 1] = a[base + 2]
            a[base + 2] = a[base + 3]
            a[base + 3] = a[base + 4]
            a[base + 4] = a[base + 5]
            a[base + 5] = a[base + 6]
            a[base + 6] = a[base + 7]
            a[base + 7] = a[base + 8]
            a[base + 8] = a[base + 9]
            a[base + 9] = a[base + 10]
            a[base + 10] = a[base + 11]
            a[base + 11] = tmp
        else:
            # General rotation using a small shared buffer (no allocations).
            for i in range(12):
                buf[i] = a[base + ((i - shift) % 12)]
            for i in range(12):
                a[base + i] = buf[i]

        v = lz[node] + shift
        lz[node] = v - 12 if v >= 12 else v

    def _push(node: int) -> None:
        s = lz[node]
        if s:
            left = node << 1
            _apply(left, s)
            _apply(left + 1, s)
            lz[node] = 0

    def _pull(node: int) -> None:
        base = node * 12
        left = node << 1
        right = left + 1
        bL = left * 12
        bR = right * 12
        for i in range(12):
            a[base + i] = a[bL + i] + a[bR + i]

    def update(node: int, l: int, r: int, ql: int, qr: int, shift: int) -> None:
        """Range add (mod 12) on y-index interval [ql, qr)."""
        if ql <= l and r <= qr:
            _apply(node, shift)
            return
        _push(node)
        mid = (l + r) >> 1
        left = node << 1
        if ql < mid:
            update(left, l, mid, ql, qr, shift)
        if qr > mid:
            update(left + 1, mid, r, ql, qr, shift)
        _pull(node)

    # Sweep along x, accumulating counts for each overlap residue.
    counts = [0] * 12  # counts[r] = number of grid points with overlap ≡ r (mod 12)
    ev_i = 0
    ev_n = len(events_idx)

    for i in range(len(x_vals) - 1):
        x = x_vals[i]
        while ev_i < ev_n and events_idx[ev_i][0] == x:
            _, sh, yl, yh1 = events_idx[ev_i]
            update(1, 0, m, yl, yh1, sh)
            ev_i += 1

        width = x_vals[i + 1] - x  # number of x points in this stripe
        if width:
            base = 12  # node 1
            counts[0] += width * a[base]
            counts[1] += width * a[base + 1]
            counts[2] += width * a[base + 2]
            counts[3] += width * a[base + 3]
            counts[4] += width * a[base + 4]
            counts[5] += width * a[base + 5]
            counts[6] += width * a[base + 6]
            counts[7] += width * a[base + 7]
            counts[8] += width * a[base + 8]
            counts[9] += width * a[base + 9]
            counts[10] += width * a[base + 10]
            counts[11] += width * a[base + 11]

    # Convert overlap residues to displayed hours:
    # residue 0 -> hour 12, residue r -> hour r (1..11).
    total = 12 * counts[0]
    for r in range(1, 12):
        total += r * counts[r]
    return total


def main() -> None:
    # Test values from the problem statement:
    assert compute_C(0) == 30621295449583788
    assert compute_C(1) == 30613048345941659
    assert compute_C(10) == 21808930308198471
    assert compute_C(100) == 16190667393984172

    print(compute_C(10**5))


if __name__ == "__main__":
    main()
