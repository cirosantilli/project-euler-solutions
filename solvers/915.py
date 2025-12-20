#!/usr/bin/env python3
"""Project Euler 915: Giant GCDs

We define
  s(1) = 1
  s(n+1) = (s(n)-1)^3 + 2

Let
  T(N) = sum_{a=1..N} sum_{b=1..N} gcd( s(s(a)), s(s(b)) )

Compute T(10^8) mod 123456789.

Constraints from the prompt:
- No external libraries.
- Assert all test values provided in the problem statement.
- Do NOT hardcode/declare the final answer; only print the computed value.

Implementation overview is in README.md.
"""

from math import gcd

MOD = 123456789


def f_mod(x: int, m: int) -> int:
    """f(x) = (x-1)^3 + 2 (mod m)."""
    y = (x - 1) % m
    y2 = (y * y) % m
    y3 = (y2 * y) % m
    return (y3 + 2) % m


def cycle_info_s_mod(m: int) -> tuple[int, int]:
    """Floyd cycle detection for s(n) modulo m.

    We consider the state sequence:
      x_0 = s(0) = 0
      x_{n+1} = f(x_n) (mod m)
    so x_n == s(n) (mod m).

    Returns (mu, lam) where the sequence becomes periodic:
      x_n = x_{mu + (n-mu) % lam} for all n >= mu.
    """

    def step(z: int) -> int:
        return f_mod(z, m)

    x0 = 0
    tortoise = step(x0)
    hare = step(step(x0))
    while tortoise != hare:
        tortoise = step(tortoise)
        hare = step(step(hare))

    mu = 0
    tortoise = x0
    while tortoise != hare:
        tortoise = step(tortoise)
        hare = step(hare)
        mu += 1

    lam = 1
    hare = step(tortoise)
    while tortoise != hare:
        hare = step(hare)
        lam += 1

    return mu, lam


def build_s_mod(m: int, length: int) -> list[int]:
    """Build s(n) mod m for n = 0..length."""
    arr = [0] * (length + 1)
    x = 0
    for n in range(1, length + 1):
        x = f_mod(x, m)
        arr[n] = x
    return arr


def sieve_phi_prefix(n: int) -> list[int]:
    """Compute prefix sums of Euler's totient function up to n."""
    phi = list(range(n + 1))
    for i in range(2, n + 1):
        if phi[i] == i:  # prime
            for j in range(i, n + 1, i):
                phi[j] -= phi[j] // i

    pref = [0] * (n + 1)
    s = 0
    for i in range(1, n + 1):
        s += phi[i]
        pref[i] = s
    return pref


def brute_T(N: int) -> int:
    """Brute-force T(N) (exact), only feasible for tiny N."""
    svals = [0]
    x = 0

    def get_s(i: int) -> int:
        nonlocal x
        while len(svals) <= i:
            x = (x - 1) ** 3 + 2
            svals.append(x)
        return svals[i]

    xs = []
    for a in range(1, N + 1):
        sa = get_s(a)
        xs.append(get_s(sa))

    tot = 0
    for i in range(N):
        for j in range(N):
            tot += gcd(xs[i], xs[j])
    return tot


def solve_T(N: int) -> int:
    """Compute T(N) mod MOD."""

    # --- periodicity of s(n) mod MOD ---
    muM, lamM = cycle_info_s_mod(MOD)
    s_mod_MOD = build_s_mod(MOD, muM + lamM)

    # To reduce indices inside the MOD-cycle we only need s(n) modulo lamM.
    muP, lamP = cycle_info_s_mod(lamM)
    s_mod_lam = build_s_mod(lamM, muP + lamP)

    # Compute exact s(n) for the very first n until s(n) > muM.
    # After that, s(n) is certainly in the index-cycle region for MOD.
    small_exact = [0]
    x = 0
    n = 0
    while True:
        n += 1
        x = (x - 1) ** 3 + 2
        small_exact.append(x)
        if x > muM:
            break
    n_small_max = n - 1

    muM_mod = muM % lamM

    def s_index_modMOD(k: int) -> int:
        """Return s(k) mod MOD for an arbitrary (possibly huge) index k."""
        if k <= muM + lamM:
            return s_mod_MOD[k]
        k2 = muM + ((k - muM) % lamM)
        return s_mod_MOD[k2]

    def s_n_mod_lamM(n_: int) -> int:
        """Return s(n_) mod lamM."""
        if n_ <= muP + lamP:
            return s_mod_lam[n_]
        n2 = muP + ((n_ - muP) % lamP)
        return s_mod_lam[n2]

    def s2_mod(n_: int) -> int:
        """Return s(s(n_)) mod MOD."""
        if n_ <= n_small_max:
            return s_index_modMOD(small_exact[n_])
        k_mod = s_n_mod_lamM(n_)
        # s(n_) is huge here, so reduce its index into the MOD-cycle using k_mod.
        idx = muM + ((k_mod - muM_mod) % lamM)
        return s_mod_MOD[idx]

    # --- s2_mod(n) becomes periodic once s(n) mod lamM is in its cycle ---
    start = max(1, n_small_max + 1, muP)
    period = lamP

    period_vals = [0] * period
    for i in range(period):
        period_vals[i] = s2_mod(start + i) % MOD

    # Prefix sums of s2 for O(1) range sums
    small_prefix = [0] * (start)
    acc = 0
    for i in range(1, start):
        acc = (acc + s2_mod(i)) % MOD
        small_prefix[i] = acc

    period_prefix = [0] * (period + 1)
    accp = 0
    for i in range(period):
        accp = (accp + period_vals[i]) % MOD
        period_prefix[i + 1] = accp
    period_sum = period_prefix[period]

    def prefix_s2(n_: int) -> int:
        """Return sum_{i=1..n_} s(s(i)) mod MOD."""
        if n_ <= 0:
            return 0
        if n_ < start:
            return small_prefix[n_]
        base = small_prefix[start - 1]
        t = n_ - (start - 1)
        full = t // period
        rem = t % period
        return (base + full * period_sum + period_prefix[rem]) % MOD

    # --- summatory totient Phi(n) = sum_{k<=n} phi(k) ---
    BASE = min(2_000_000, N)
    phi_prefix = sieve_phi_prefix(BASE)
    memo: dict[int, int] = {}

    def phi_sum(n_: int) -> int:
        if n_ <= BASE:
            return phi_prefix[n_]
        if n_ in memo:
            return memo[n_]
        res = n_ * (n_ + 1) // 2
        l = 2
        while l <= n_:
            q = n_ // l
            r = n_ // q
            res -= (r - l + 1) * phi_sum(q)
            l = r + 1
        memo[n_] = res
        return res

    def coprime_pairs(m_: int) -> int:
        # C(m) = 2*Phi(m) - 1
        return (2 * phi_sum(m_) - 1) % MOD

    # --- block over d where floor(N/d) is constant ---
    ans = 0
    l = 1
    while l <= N:
        q = N // l
        r = N // q
        sum_s2 = (prefix_s2(r) - prefix_s2(l - 1)) % MOD
        ans = (ans + sum_s2 * coprime_pairs(q)) % MOD
        l = r + 1

    return ans % MOD


def main() -> None:
    # Asserts for test values from the problem statement
    assert solve_T(3) == 12
    assert solve_T(4) == (24881925 % MOD)
    assert solve_T(100) == (14416749 % MOD)

    # extra sanity check (tiny brute)
    assert brute_T(3) == 12

    print(solve_T(10**8))


if __name__ == "__main__":
    main()
