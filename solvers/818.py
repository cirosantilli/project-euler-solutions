#!/usr/bin/env python3
"""
Project Euler 818: SET

Compute F(12) where
  F(n) = sum_{|C|=n} S(C)^4
and S(C) is the number of SETs (affine lines) contained in C.

No external libraries are used.
"""

from itertools import combinations


def nCk(n: int, k: int) -> int:
    """Binomial coefficient (exact, integer)."""
    if k < 0 or k > n:
        return 0
    k = k if k <= n - k else n - k
    res = 1
    # res = product_{i=1..k} (n-k+i)/i
    for i in range(1, k + 1):
        res = (res * (n - k + i)) // i
    return res


def build_geometry():
    """
    Model the SET deck as the affine space F_3^4 (81 points).
    A SET corresponds to an affine line of 3 points.

    Returns:
      line_masks: list[int] of length 1080, 81-bit masks for each line
      line_dirs:  list[int] directions (0..39) for each line
    """
    # Precompute base-3 digits for point ids 0..80 (4 digits)
    d0 = [0] * 81
    d1 = [0] * 81
    d2 = [0] * 81
    d3 = [0] * 81
    for pid in range(81):
        x = pid
        d0[pid] = x % 3
        x //= 3
        d1[pid] = x % 3
        x //= 3
        d2[pid] = x % 3
        x //= 3
        d3[pid] = x % 3

    def third_point(i: int, j: int) -> int:
        # c = -i - j in F_3^4, coordinate-wise
        a0 = (d0[i] + d0[j]) % 3
        a1 = (d1[i] + d1[j]) % 3
        a2 = (d2[i] + d2[j]) % 3
        a3 = (d3[i] + d3[j]) % 3
        c0 = (-a0) % 3
        c1 = (-a1) % 3
        c2 = (-a2) % 3
        c3 = (-a3) % 3
        return c0 + 3 * c1 + 9 * c2 + 27 * c3

    # Direction vectors are nonzero in F_3^4, modulo multiplication by 2 (i.e., v ~ -v).
    # Represent a vector by its base-3 digits packed into an int in [0..80].
    def vec_key(v0, v1, v2, v3) -> int:
        return v0 + 3 * v1 + 9 * v2 + 27 * v3

    neg_key = [0] * 81
    for key in range(81):
        # -v == 2*v mod 3
        v0 = (2 * (key % 3)) % 3
        v1 = (2 * ((key // 3) % 3)) % 3
        v2 = (2 * ((key // 9) % 3)) % 3
        v3 = (2 * ((key // 27) % 3)) % 3
        neg_key[key] = vec_key(v0, v1, v2, v3)

    dir_index = {}
    dirs = []
    for key in range(1, 81):
        canon = key if key < neg_key[key] else neg_key[key]
        if canon not in dir_index:
            dir_index[canon] = len(dirs)
            dirs.append(canon)
    assert len(dirs) == 40  # number of directions in F_3^4

    # Enumerate all lines from all unordered point pairs, deduplicated
    line_triples = set()
    for i, j in combinations(range(81), 2):
        k = third_point(i, j)
        if k == i or k == j:
            continue
        a, b, c = sorted((i, j, k))
        line_triples.add((a, b, c))
    assert len(line_triples) == 1080

    line_triples = sorted(line_triples)
    line_masks = [0] * 1080
    line_dirs = [0] * 1080

    for idx, (a, b, c) in enumerate(line_triples):
        line_masks[idx] = (1 << a) | (1 << b) | (1 << c)

        # Direction from (a -> b), canonicalized by v ~ -v
        v0 = (d0[b] - d0[a]) % 3
        v1 = (d1[b] - d1[a]) % 3
        v2 = (d2[b] - d2[a]) % 3
        v3 = (d3[b] - d3[a]) % 3
        key = vec_key(v0, v1, v2, v3)
        if key == 0:
            # fallback shouldn't happen, but keep robust
            v0 = (d0[c] - d0[a]) % 3
            v1 = (d1[c] - d1[a]) % 3
            v2 = (d2[c] - d2[a]) % 3
            v3 = (d3[c] - d3[a]) % 3
            key = vec_key(v0, v1, v2, v3)
        canon = key if key < neg_key[key] else neg_key[key]
        line_dirs[idx] = dir_index[canon]

    return line_masks, line_dirs


def compute_Ak(line_masks, line_dirs):
    """
    Let L be the set of all 1080 lines. Expanding S(C)^4 produces ordered 4-tuples of lines.
    Group ordered *pairs* of lines (l1,l2) into 4 symmetry classes and use a representative of
    each class to count intersections with every other class.

    Returns A_k for k=0..12, where A_k is the number of ordered 4-tuples of lines whose union
    covers exactly k distinct cards/points.
    """
    L = len(line_masks)
    assert L == 1080

    # Classes of ordered line pairs (i,j):
    # 0: i==j (same line)                  union size 3
    # 1: intersecting distinct lines       union size 5
    # 2: disjoint parallel distinct lines  union size 6
    # 3: disjoint non-parallel (skew)      union size 6
    pair_masks = [[] for _ in range(4)]
    masks = line_masks
    dirs = line_dirs

    for i in range(L):
        mi = masks[i]
        di = dirs[i]
        for j in range(L):
            if i == j:
                pair_masks[0].append(mi)
                continue
            mj = masks[j]
            if mi & mj:
                pair_masks[1].append(mi | mj)
            else:
                if di == dirs[j]:
                    pair_masks[2].append(mi | mj)
                else:
                    pair_masks[3].append(mi | mj)

    sizes = [3, 5, 6, 6]
    reps = [pair_masks[c][0] for c in range(4)]

    # A_k for k up to 12
    A_k = [0] * 13

    for a in range(4):
        repA = reps[a]
        countA = len(pair_masks[a])
        sizeA = sizes[a]

        for b in range(4):
            sizeB = sizes[b]
            dist = [0] * 7  # intersection size r in 0..6
            for m in pair_masks[b]:
                dist[(repA & m).bit_count()] += 1
            for r, cnt in enumerate(dist):
                if cnt:
                    k = sizeA + sizeB - r
                    A_k[k] += countA * cnt

    # Sanity: total ordered 4-tuples must be 1080^4
    assert sum(A_k) == 1080**4
    # Only union sizes 3..12 are possible
    assert all(A_k[k] == 0 for k in range(3))
    return A_k


def F_from_Ak(A_k, n: int) -> int:
    tot = 0
    for k in range(0, n + 1):
        if A_k[k]:
            tot += A_k[k] * nCk(81 - k, n - k)
    return tot


def main():
    line_masks, line_dirs = build_geometry()
    A_k = compute_Ak(line_masks, line_dirs)

    # Test values from the problem statement
    assert F_from_Ak(A_k, 3) == 1080
    assert F_from_Ak(A_k, 6) == 159690960

    print(F_from_Ak(A_k, 12))


if __name__ == "__main__":
    main()
