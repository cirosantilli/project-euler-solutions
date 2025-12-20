#!/usr/bin/env python3
"""
Project Euler 933 - Paper Cutting

Two-player impartial game on integer rectangles:
A move picks one rectangle and cuts it once vertically and once horizontally
to produce four smaller rectangles (all sides positive integers).
No move => lose.

Let C(w,h) be number of winning first moves from a w x h rectangle.
Let D(W,H) = sum_{w=2..W} sum_{h=2..H} C(w,h).

We compute D(123, 1234567).

Approach:
1) Compute Sprague-Grundy numbers G(w,h) (nimbers), using mex of reachable XOR outcomes.
2) For each fixed width w, prove that G(w,h) becomes constant beyond some index s
   by detecting that it is constant on the interval [s, 2s].
3) Count D(W,H) via quadruple counting:
   A move corresponds to choosing (i,j,k,l) with i,j>=1, k,l>=1,
   i+j<=W, k+l<=H, and winning iff:
       G(i,k) xor G(j,k) == G(i,l) xor G(j,l)

No external libraries are used (only Python standard library).
"""

from bisect import bisect_right


class GrundyComputer:
    """
    Incrementally computes and stores:
      - G[w][h] for h up to a computed limit B[w]
      - g_const[w] = s where G(w,h) is constant for all h >= s
      - const[w] = that constant nimber value
    """

    def __init__(self):
        self.current_w = 1
        self.B = [0, 1]  # B[w] = computed height limit for width w
        self.g_const = [0, 1]  # stabilization start index for width w
        self.const = [0, 0]  # constant value after stabilization
        self.G = [None, [0, 0]]  # G[w] is list indexed by height
        self.visited = [0] * 32768
        self.stamp = 1
        self.prev_maxg = 1

    def _ensure_size(self, n: int) -> None:
        need = n + 1 - len(self.B)
        if need > 0:
            self.B.extend([0] * need)
            self.g_const.extend([0] * need)
            self.const.extend([0] * need)
            self.G.extend([None] * need)

    def getG(self, wi: int, hi: int) -> int:
        """Return G(min(wi,hi), max(wi,hi)), using const beyond stored limit."""
        if wi > hi:
            wi, hi = hi, wi
        if wi <= 1:
            return 0
        lim = self.B[wi]
        if hi <= lim:
            return self.G[wi][hi]
        return self.const[wi]

    def compute_next(self) -> None:
        """Compute data for the next width (current_w + 1)."""
        w = self.current_w + 1
        self._ensure_size(w)

        # Initial working limit; may be extended if stability proof needs more range.
        limit = max(w, 2 * self.prev_maxg)

        getG = self.getG

        # g_w[0] unused; g_w[h] holds nimber for w x h (with w<=h implied by getG symmetry)
        g_w = [0] * (limit + 1)

        # Fill h < w by symmetry (w x h == h x w), using const if necessary
        upto = min(w, limit + 1)
        for h in range(1, upto):
            g_w[h] = getG(h, w)

        half_w = w >> 1

        # For each vertical split a (1..floor(w/2)), precompute
        # V_a[b] = G(a,b) xor G(w-a,b), plus threshold ta where both parts stabilized.
        Vs = []
        t_as = []
        for a in range(1, half_w + 1):
            bw = w - a
            ta = (
                self.g_const[a]
                if self.g_const[a] > self.g_const[bw]
                else self.g_const[bw]
            )
            t_as.append(ta)
            V = [0] * (limit + 1)
            for b in range(1, limit + 1):
                V[b] = getG(a, b) ^ getG(bw, b)
            Vs.append(V)

        vis = self.visited
        s = self.stamp

        def compute_h_range(h_start: int, h_end: int) -> None:
            """Compute nimbers g_w[h] for h in [h_start..h_end]."""
            nonlocal s, vis, g_w
            for h in range(h_start, h_end + 1):
                s += 1
                half_h = h >> 1
                hh = h
                for V, ta in zip(Vs, t_as):
                    # If half_h >= ta then b in [ta..half_h] gives 0 outcomes,
                    # so 0 is reachable; otherwise only prefix pairs matter.
                    if half_h >= ta:
                        vis[0] = s
                        maxb = ta - 1
                        if maxb > half_h:
                            maxb = half_h
                    else:
                        maxb = half_h

                    V_local = V
                    vis_local = vis
                    for b in range(1, maxb + 1):
                        v = V_local[b] ^ V_local[hh - b]
                        # mex values stay small; large visited table avoids frequent resizing
                        if v < 32768:
                            vis_local[v] = s
                        else:
                            # Rare fallback: expand visited if needed
                            n = len(vis_local)
                            while n <= v:
                                n *= 2
                            vis_local.extend([0] * (n - len(vis_local)))
                            vis_local[v] = s
                            vis = vis_local
                            self.visited = vis_local

                # mex
                m = 0
                while vis[m] == s:
                    m += 1
                g_w[h] = m

        # Compute initial range
        compute_h_range(w, limit)

        # Track last change index
        last = 1
        for h in range(2, limit + 1):
            if g_w[h] != g_w[h - 1]:
                last = h

        # Extend until we can prove stability:
        # if constant on [last, 2*last], then constant forever.
        while True:
            need = 2 * last
            if need <= limit:
                v0 = g_w[last]
                ok = True
                for hh in range(last + 1, need + 1):
                    if g_w[hh] != v0:
                        ok = False
                        break
                if ok:
                    break

            new_limit = max(limit * 2, need, w)

            # Extend g_w
            g_w.extend([0] * (new_limit - limit))

            # Extend V arrays
            for idx, a in enumerate(range(1, half_w + 1)):
                bw = w - a
                V = Vs[idx]
                V.extend([0] * (new_limit - limit))
                for b in range(limit + 1, new_limit + 1):
                    V[b] = getG(a, b) ^ getG(bw, b)

            # Compute added heights
            compute_h_range(limit + 1, new_limit)
            limit = new_limit

            # Update last change using newly computed part
            for h in range(last + 1, limit + 1):
                if g_w[h] != g_w[h - 1]:
                    last = h

        # Store results for this width
        self.B[w] = limit
        self.g_const[w] = last
        self.const[w] = g_w[last]
        self.G[w] = g_w
        self.stamp = s
        self.current_w = w
        if last > self.prev_maxg:
            self.prev_maxg = last

    def compute_upto(self, W: int) -> None:
        while self.current_w < W:
            self.compute_next()


def count_winning_moves_C(w: int, h: int, getG_func) -> int:
    """Directly count winning moves for a single rectangle w x h."""
    cnt = 0
    for x in range(1, w):
        for y in range(1, h):
            v = (
                getG_func(x, y)
                ^ getG_func(w - x, y)
                ^ getG_func(x, h - y)
                ^ getG_func(w - x, h - y)
            )
            if v == 0:
                cnt += 1
    return cnt


def compute_D(W: int, H: int, gc: GrundyComputer) -> int:
    """
    Compute D(W,H) via quadruple counting over ordered splits (i,j) and (k,l).

    For ordered (i,j) with i+j<=W, define t(k)=G(i,k) xor G(j,k).
    We count ordered (k,l) with k+l<=H and t(k)=t(l).
    """
    g_const = gc.g_const
    const = gc.const

    max_m = 0
    for w in range(1, W + 1):
        if g_const[w] > max_m:
            max_m = g_const[w]

    # Precompute G(u,k) for all u<=W and k<=min(max_m,H-1) for speed
    max_k = min(max_m, H - 1)
    row = [[0] * (max_k + 1) for _ in range(W + 1)]
    for u in range(1, W + 1):
        for k in range(1, max_k + 1):
            row[u][k] = gc.getG(u, k)

    total = 0

    for i in range(1, W):
        ri = row[i]
        for j in range(1, W - i + 1):  # ordered (i,j)
            rj = row[j]
            m = g_const[i] if g_const[i] > g_const[j] else g_const[j]
            const_t = const[i] ^ const[j]

            # Fast branch when H is large enough that all prefix-prefix pairs fit under k+l<=H
            if H >= 2 * m:
                L = m - 1

                if L > 0:
                    cnts = {}
                    count_eq = 0
                    sum_k_eq = 0
                    for k in range(1, L + 1):
                        t = ri[k] ^ rj[k]
                        cnts[t] = cnts.get(t, 0) + 1
                        if t == const_t:
                            count_eq += 1
                            sum_k_eq += k

                    countA = 0
                    for c in cnts.values():
                        countA += c * c
                else:
                    countA = 0
                    count_eq = 0
                    sum_k_eq = 0

                # prefix-tail and tail-prefix
                countB = 0
                if count_eq:
                    # sum_{k in prefix, t(k)=const_t} (H-k - m + 1)
                    countB = 2 * (count_eq * (H - m + 1) - sum_k_eq)

                # tail-tail triangle: k,l >= m and k+l <= H
                S = H - 2 * m
                countC = (S + 1) * (S + 2) // 2 if S >= 0 else 0

                total += countA + countB + countC

            else:
                # Generic smaller-H branch (used mainly for the provided test)
                pos = {}
                for k in range(1, H):
                    t = (
                        (ri[k] ^ rj[k])
                        if k <= max_k
                        else (gc.getG(i, k) ^ gc.getG(j, k))
                    )
                    pos.setdefault(t, []).append(k)

                for lst in pos.values():
                    lst.sort()
                    for k in lst:
                        total += bisect_right(lst, H - k)

    return total


def main() -> None:
    W = 123
    H = 1234567

    gc = GrundyComputer()
    gc.compute_upto(W)

    # Test assertions from problem statement
    assert count_winning_moves_C(5, 3, gc.getG) == 4
    assert compute_D(12, 123, gc) == 327398

    # Final answer (must not be embedded as a constant in the file)
    print(compute_D(W, H, gc))


if __name__ == "__main__":
    main()
