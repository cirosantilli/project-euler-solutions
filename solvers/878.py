#!/usr/bin/env python3
"""
Project Euler 878: XOR-Equation B

We use:
  - x ^ y  : bitwise XOR
  - x ⊗ y  : carryless (GF(2)) multiplication ("XOR-product")

We count pairs (a,b) with 0 <= a <= b <= N such that

    (a ⊗ a) ^ (2 ⊗ a ⊗ b) ^ (b ⊗ b) <= m

No external libraries are used.
"""


def clmul(x: int, y: int) -> int:
    """
    Carryless multiplication over GF(2).

    If x = sum x_i 2^i and y = sum y_j 2^j (x_i,y_j in {0,1}),
    then:
        x ⊗ y = XOR over all i,j with x_i=y_j=1 of 2^(i+j)

    Implementation: iterate over set bits of the sparser operand.
    """
    if x == 0 or y == 0:
        return 0
    # iterate over fewer set bits for speed
    if x.bit_count() < y.bit_count():
        x, y = y, x
    res = 0
    while y:
        lsb = y & -y
        res ^= x << (lsb.bit_length() - 1)
        y ^= lsb
    return res


def value_k(a: int, b: int) -> int:
    """
    k(a,b) = (a⊗a) ^ (2⊗a⊗b) ^ (b⊗b)

    Since ⊗ is associative/commutative and 2 ⊗ z == z<<1:
      2⊗a⊗b == (a⊗b)<<1
    """
    return clmul(a, a) ^ (clmul(a, b) << 1) ^ clmul(b, b)


def T(a: int, b: int) -> tuple[int, int]:
    """
    Orbit step (unit action):
      (a,b) -> (b, a ^ (b<<1))
    """
    return b, a ^ (b << 1)


def T_inv(a: int, b: int) -> tuple[int, int]:
    """
    Inverse orbit step:
      (a,b) -> (b ^ (a<<1), a)
    """
    return b ^ (a << 1), a


def box_size_for_m(m: int) -> int:
    """
    Empirically/theoretically, all solution-orbits for k <= m intersect
    a small "fundamental box" whose side length is about 2^(deg(m)/2).

    We take:
      B = 2^(((bitlen(m)+1)//2) + 2)

    For m=1_000_000 (bitlen=20) => B=2^12=4096.
    """
    return 1 << (((m.bit_length() + 1) // 2) + 2)


def collect_solutions_in_box(m: int) -> tuple[int, list[int], set[int]]:
    """
    Enumerate all (a,b) with 0 <= a,b < B that satisfy k(a,b) <= m.

    We enumerate only a<=b, but insert both orientations (a,b) and (b,a)
    so orbit connectivity inside the box is preserved.

    Returns:
      B, precomputed squares sq[x]=x⊗x for x<B, and a set of encoded pairs.
    """
    B = box_size_for_m(m)

    sq = [0] * B
    for x in range(B):
        sq[x] = clmul(x, x)

    sols = set()
    for a in range(B):
        sa = sq[a]
        for b in range(a, B):
            k = sa ^ sq[b] ^ (clmul(a, b) << 1)
            if k <= m:
                sols.add(a * B + b)
                if a != b:
                    sols.add(b * B + a)

    return B, sq, sols


def orbits_from_box(
    B: int, sq: list[int], sols: set[int]
) -> list[tuple[int, int, int]]:
    """
    Find connected components inside the box under T and T_inv.

    Each component corresponds to (part of) a full orbit; later we merge
    by a global canonical representative.

    Returns list of (a,b,k) representatives (a,b within the box).
    """
    reps = []
    visited = set()

    def in_box(x: int) -> bool:
        return 0 <= x < B

    for code in sols:
        if code in visited:
            continue

        stack = [code]
        comp = []

        while stack:
            c = stack.pop()
            if c in visited or c not in sols:
                continue
            visited.add(c)
            comp.append(c)

            a = c // B
            b = c - a * B

            na, nb = T(a, b)
            if in_box(na) and in_box(nb):
                stack.append(na * B + nb)

            na, nb = T_inv(a, b)
            if in_box(na) and in_box(nb):
                stack.append(na * B + nb)

        rep_code = min(comp)
        ra = rep_code // B
        rb = rep_code - ra * B
        k = sq[ra] ^ sq[rb] ^ (clmul(ra, rb) << 1)
        reps.append((ra, rb, k))

    return reps


def canonical_pair(a: int, b: int, steps: int = 120) -> tuple[int, int]:
    """
    Produce a stable orbit key by exploring a bounded window of the orbit
    in both directions and taking the pair with minimal (b,a) lexicographically.

    This is enough to merge any duplicates coming from 'box-local' components.
    """
    best = (b, a)
    best_pair = (a, b)

    seen = set()

    x, y = a, b
    for _ in range(steps):
        if (y, x) < best:
            best = (y, x)
            best_pair = (x, y)
        x, y = T(x, y)
        if (x, y) in seen:
            break
        seen.add((x, y))

    x, y = a, b
    for _ in range(steps):
        x, y = T_inv(x, y)
        if (y, x) < best:
            best = (y, x)
            best_pair = (x, y)
        if (x, y) in seen:
            break
        seen.add((x, y))

    return best_pair


def count_orbit_solutions(a0: int, b0: int, N: int, max_steps: int) -> int:
    """
    Count distinct pairs in the orbit of (a0,b0) that satisfy 0<=a<=b<=N.

    We walk in both directions for a bounded number of steps; for this problem,
    orbits grow in bit-length quickly, so max_steps ~ bitlen(N) + constant is ample.
    """
    total = 0
    seen = set()

    for step_fn in (T, T_inv):
        a, b = a0, b0
        for _ in range(max_steps):
            if (a, b) in seen:
                break
            seen.add((a, b))
            if a <= b and b <= N:
                total += 1
            a, b = step_fn(a, b)

    return total


def G(N: int, m: int) -> int:
    """
    Compute G(N,m): number of solutions with k<=m and 0<=a<=b<=N.
    """
    B, sq, sols = collect_solutions_in_box(m)
    box_reps = orbits_from_box(B, sq, sols)

    # Merge by canonical representative
    uniq = {}
    for a, b, k in box_reps:
        ca, cb = canonical_pair(a, b)
        key = (ca, cb)
        if key not in uniq:
            uniq[key] = (ca, cb, k)

    max_steps = N.bit_length() + 80

    total = 0
    for a, b, _k in uniq.values():
        total += count_orbit_solutions(a, b, N, max_steps)

    return total


def _self_tests() -> None:
    # From the statement: 7 ⊗ 3 = 9
    assert clmul(7, 3) == 9

    # From the statement: (a,b)=(3,6) is a solution for k=5
    assert value_k(3, 6) == 5

    # From the statement: G(1000,100)=398
    assert G(1000, 100) == 398


def main() -> None:
    _self_tests()
    N = 10**17
    m = 1_000_000
    print(G(N, m))


if __name__ == "__main__":
    main()
