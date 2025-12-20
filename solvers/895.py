#!/usr/bin/env python3
"""
Project Euler 895 — Gold & Silver Coin Game II

We count ordered triples of stacks (A,B,C), each a non-empty stack of up to m coins,
that are both:
  * fair:    G(A)+G(B)+G(C) = 0   (game values sum to 0)
  * balanced D(A)+D(B)+D(C) = 0   (#gold - #silver sums to 0)

Key facts (derived in README):
  * Every stack is either monochrome (integer-valued) or mixed (dyadic rational).
  * Mixed stacks can be parameterized by:
        value = k + (2x+1)/2^t
        diff  = k + offset + w
    where w = 2*popcount(x) - (t-1), offset=0 for gold-start, 1 for silver-start.
  * For 3 mixed stacks, fairness constrains denominators to pattern (T,T,u), u<T.

We compute G(9898) mod 989898989 without external libraries.

The final numeric answer is not hard-coded anywhere; it is printed.
"""

# -------------------------
# Small utilities
# -------------------------


def egcd(a, b):
    while b:
        a, b = b, a % b
    return a


def modinv(a, m):
    # extended gcd
    x0, x1 = 1, 0
    y0, y1 = 0, 1
    aa, bb = a, m
    while bb:
        q = aa // bb
        aa, bb = bb, aa - q * bb
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    if aa != 1:
        raise ValueError("inverse does not exist")
    return x0 % m


def ceil_div(n, d):
    # d>0
    return -((-n) // d)


# choose(S+2,2) for S>=0 else 0, computed as integer
def tri(S):
    if S < 0:
        return 0
    t = S + 2
    return t * (t - 1) // 2


def bounded_sum_count(lengths, target):
    """
    Count integer solutions y_i in [0, n_i-1] such that sum y_i = target.
    lengths: list of 3 positive lengths
    """
    n0, n1, n2 = lengths
    S = target
    total = 0
    # inclusion-exclusion over which variables exceed upper bound
    # mask 0..7
    for mask in range(8):
        sub = 0
        bits = 0
        if mask & 1:
            sub += n0
            bits += 1
        if mask & 2:
            sub += n1
            bits += 1
        if mask & 4:
            sub += n2
            bits += 1
        val = tri(S - sub)
        if bits & 1:
            total -= val
        else:
            total += val
    return total


# -------------------------
# Exact (small m) solver
# -------------------------


def G_exact(m):
    """
    Exact integer computation for small m (used for asserts).
    This uses the same derived formulas but keeps straightforward loops.
    """
    # Case 0: all 3 monochrome
    case0 = 3 * m * (m - 1)

    # Case 2: two mixed + one monochrome
    case2 = 0
    for t in range(1, m):
        n = m - t
        case2 += (1 << (t - 1)) * n * (n - 1)
    case2 *= 6

    # Case 3: three mixed
    case3 = 0

    # carry DP for u-bit core counting:
    # cur0[c] / cur1[c] = weighted #carry sequences length u ending in carry 0/1
    # with internal carry-ones count = c.
    cur0 = [1]  # u=1
    cur1 = [1]

    for u in range(1, m - 1):  # u <= m-2 is relevant, but keep update clean
        if u <= m - 2:
            for T in range(u + 1, m):
                L = T - u - 1
                factor_low = 1 << L

                for s in (1, 2):
                    f = s - 1
                    for placement in range(3):
                        # exponents per position:
                        # two are T, one is u
                        tpos = [T, T, T]
                        tpos[placement] = u

                        # sign patterns: 0=gold-start (positive base interval), 1=silver-start (negative)
                        for mask in range(8):
                            r = (mask & 1) + ((mask >> 1) & 1) + ((mask >> 2) & 1)
                            Wtarget = s - r
                            num = Wtarget + u + 1 - 4 * f
                            if num & 1:
                                continue
                            C = num // 2
                            if not (0 <= C <= u - 1):
                                continue
                            numerator_high = cur0[C] if f == 0 else cur1[C]

                            # base sum constraint: k1+k2+k3 = -s with ranges depending on sign and t
                            intervals = []
                            for i in range(3):
                                t_i = tpos[i]
                                n_i = m - t_i
                                if ((mask >> i) & 1) == 0:
                                    # positive: [0..n_i-1]
                                    intervals.append((0, n_i))
                                else:
                                    # negative: [-n_i..-1]
                                    intervals.append((-n_i, n_i))
                            Lsum = intervals[0][0] + intervals[1][0] + intervals[2][0]
                            lens = [intervals[0][1], intervals[1][1], intervals[2][1]]
                            S = (-s) - Lsum
                            base_count = bounded_sum_count(lens, S)
                            if base_count:
                                case3 += numerator_high * factor_low * base_count

        # update DP to u+1
        nxt0 = [0] * (u + 1)
        nxt1 = [0] * (u + 1)
        # c=0
        nxt0[0] = 3 * cur0[0]
        nxt1[0] = cur0[0]
        for c in range(1, u):
            nxt0[c] = 3 * cur0[c] + cur1[c - 1]
            nxt1[c] = cur0[c] + 3 * cur1[c - 1]
        # c=u
        nxt0[u] = cur1[u - 1]
        nxt1[u] = 3 * cur1[u - 1]
        cur0, cur1 = nxt0, nxt1

    return case0 + case2 + case3


# -------------------------
# Fast modular solver
# -------------------------


def G_mod(m, MOD):
    inv2 = modinv(2, MOD)

    # precompute pow2 and invpow2 up to m
    pow2 = [1] * (m + 1)
    for i in range(1, m + 1):
        pow2[i] = (pow2[i - 1] * 2) % MOD

    invpow2 = [1] * (m + 1)
    invpow2[1] = inv2 % MOD
    for i in range(2, m + 1):
        invpow2[i] = (invpow2[i - 1] * inv2) % MOD

    # prefix sums of invpow2, a*invpow2, a^2*invpow2
    P0 = [0] * (m + 1)
    P1 = [0] * (m + 1)
    P2 = [0] * (m + 1)
    for a in range(1, m + 1):
        w = invpow2[a]
        P0[a] = (P0[a - 1] + w) % MOD
        P1[a] = (P1[a - 1] + a * w) % MOD
        P2[a] = (P2[a - 1] + (a * a) * w) % MOD

    def interval_sums(l, r):
        if l > r:
            return 0, 0, 0
        s0 = (P0[r] - P0[l - 1]) % MOD
        s1 = (P1[r] - P1[l - 1]) % MOD
        s2 = (P2[r] - P2[l - 1]) % MOD
        return s0, s1, s2

    def sum_F_linear(alpha, beta, l, r):
        # sum_{a=l..r} F(alpha*a+beta) * invpow2[a]  where F(x)=C(x+2,2)
        if l > r:
            return 0
        s0, s1, s2 = interval_sums(l, r)
        a_mod = alpha % MOD
        b_mod = beta % MOD

        # inv2*(alpha^2*S2 + alpha*(2beta+3)*S1 + (beta^2+3beta+2)*S0)
        term2 = (a_mod * a_mod) % MOD
        term1 = (a_mod * ((2 * b_mod + 3) % MOD)) % MOD
        term0 = (b_mod * b_mod + 3 * b_mod + 2) % MOD

        res = (term2 * s2 + term1 * s1 + term0 * s0) % MOD
        return (res * inv2) % MOD

    C2 = [1, 2, 1]  # C(2,ca)

    def G_pq(b, s, p, q):
        """
        G_pq = sum_{a=1..b-1} H(a)*invpow2[a] where
          H(a) = #solutions y1+y2+y3 = S with bounds (a,a,b)
          S = p*a + q*b - s
        computed via inclusion-exclusion over exceeding bounds.
        """
        Amax = b - 1
        total = 0
        for ca in (0, 1, 2):
            mult = C2[ca]
            for cb in (0, 1):
                sign = -1 if ((ca + cb) & 1) else 1
                coeff = sign * mult
                alpha = p - ca
                beta = (q - cb) * b - s

                # Determine active interval where alpha*a + beta >= 0
                if alpha == 0:
                    if beta < 0:
                        continue
                    l, r = 1, Amax
                elif alpha > 0:
                    l = max(1, ceil_div(-beta, alpha))
                    r = Amax
                    if l > r:
                        continue
                else:
                    r = min(Amax, beta // (-alpha))
                    l = 1
                    if r < l:
                        continue

                val = sum_F_linear(alpha, beta, l, r)
                total = (total + coeff * val) % MOD
        return total

    def base_weighted(b, s):
        """
        For fixed u (via b=m-u) and fixed fractional-sum s in {1,2},
        compute base-weighted sums grouped by r = #negative stacks:

          base[r] = sum_{T=u+1..m-1} 2^{T-u-1} * sum_{placements,signs with r negatives} BaseCount(...)

        We have eliminated the T loop by substituting a=m-T and weighting by invpow2[a].
        """
        # precompute all G_pq for p in 0..2, q in 0..1
        G = [[0, 0], [0, 0], [0, 0]]
        for p in (0, 1, 2):
            for q in (0, 1):
                G[p][q] = G_pq(b, s, p, q)

        base = [0, 0, 0, 0]
        # r = total negatives
        # nb = 1 if the b-length stack is negative, else 0
        # ra = r-nb negatives among the two a-length stacks
        for r in (0, 1, 2, 3):
            acc = 0
            for nb in (0, 1):
                ra = r - nb
                if 0 <= ra <= 2:
                    # multiplicity: choose which position is the b stack (3),
                    # and choose which of the 2 a positions are negative (C(2,ra)).
                    mult_sign = 3 * C2[ra]
                    acc = (acc + mult_sign * G[ra][nb]) % MOD
            # restore factor 2^{b-1}
            base[r] = (acc * pow2[b - 1]) % MOD
        return base

    # Case 0: 3 monochrome
    case0 = (3 * m * (m - 1)) % MOD

    # Case 2: two mixed + one monochrome
    case2 = 0
    for t in range(1, m):
        n = m - t
        term = (pow2[t - 1] * n * (n - 1)) % MOD
        case2 = (case2 + term) % MOD
    case2 = (case2 * 6) % MOD

    # Case 3: three mixed
    case3 = 0

    # carry DP for u-bit core:
    cur0 = [1]  # u=1
    cur1 = [1]

    for u in range(1, m - 1):  # up to m-2 used
        if u <= m - 2:
            b = m - u

            # base-weighted grouped by r for s=1 and s=2
            base_s1 = base_weighted(b, 1)
            base_s2 = base_weighted(b, 2)

            # add contributions for s=1 (f=0) and s=2 (f=1)
            # Only r>=1 gives any base solutions; r=0 yields 0 automatically.
            for s, f, base in ((1, 0, base_s1), (2, 1, base_s2)):
                for r in (1, 2, 3):
                    Wtarget = s - r
                    num = Wtarget + u + 1 - 4 * f
                    if num & 1:
                        continue
                    C = num // 2
                    if not (0 <= C <= u - 1):
                        continue
                    numerator_high = cur0[C] if f == 0 else cur1[C]
                    case3 = (case3 + numerator_high * base[r]) % MOD

        # update DP to u+1 (mod MOD)
        nxt0 = [0] * (u + 1)
        nxt1 = [0] * (u + 1)
        nxt0[0] = (3 * cur0[0]) % MOD
        nxt1[0] = cur0[0] % MOD
        for c in range(1, u):
            nxt0[c] = (3 * cur0[c] + cur1[c - 1]) % MOD
            nxt1[c] = (cur0[c] + 3 * cur1[c - 1]) % MOD
        nxt0[u] = cur1[u - 1] % MOD
        nxt1[u] = (3 * cur1[u - 1]) % MOD
        cur0, cur1 = nxt0, nxt1

    return (case0 + case2 + case3) % MOD


# -------------------------
# Main / tests
# -------------------------


def main():
    # problem statement tests (exact)
    assert G_exact(2) == 6
    assert G_exact(5) == 348
    assert G_exact(20) == 125825982708

    m = 9898
    MOD = 989898989
    ans = G_mod(m, MOD)
    print(ans)


if __name__ == "__main__":
    main()
