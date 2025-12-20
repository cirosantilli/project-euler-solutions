#!/usr/bin/env python3
"""Project Euler 851: SOP and POS

We need R_6(10000!) mod 1e9+7.

Key observation:
  For one coordinate, sum_{a,b>=1, ab=m} (a+b) = 2*sum_{d|m} d = 2*sigma_1(m).
Thus the 2D generating function factorizes and
  R_n(M) is the coefficient of q^M in (F(q))^n,
  where F(q) = sum_{m>=1} 2*sigma_1(m) q^m.

Using the (quasi)modular form E2:
  E2(q) = 1 - 24*sum_{m>=1} sigma_1(m) q^m,
so
  F(q) = (1 - E2(q))/12.
Therefore
  R_n(M) = [q^M] (1 - E2(q))^n / 12^n.

We compute the needed coefficient via expressing powers E2^k in a basis of
Eisenstein series and q-derivatives (Ramanujan differential system), plus the
cusp form Delta when needed.

No external libraries are used.
"""

MOD = 1_000_000_007


# --------------------------- basic number theory ---------------------------


def sieve(limit: int) -> list[int]:
    """Simple sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = bytearray(b"\x01") * (limit + 1)
    is_prime[0:2] = b"\x00\x00"
    p = 2
    while p * p <= limit:
        if is_prime[p]:
            step = p
            start = p * p
            is_prime[start : limit + 1 : step] = b"\x00" * (
                ((limit - start) // step) + 1
            )
        p += 1
    return [i for i in range(2, limit + 1) if is_prime[i]]


def factorize_small(n: int, primes: list[int]) -> dict[int, int]:
    """Prime factorization for n <= 10^9-ish using provided primes."""
    x = n
    out: dict[int, int] = {}
    for p in primes:
        if p * p > x:
            break
        if x % p == 0:
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            out[p] = e
    if x > 1:
        out[x] = out.get(x, 0) + 1
    return out


def factorial_prime_exponents(n: int, primes: list[int]) -> dict[int, int]:
    """Return {p: v_p(n!)} for primes p<=n."""
    exps: dict[int, int] = {}
    for p in primes:
        if p > n:
            break
        e = 0
        t = n
        while t:
            t //= p
            e += t
        if e:
            exps[p] = e
    return exps


def inv_table(n: int, mod: int) -> list[int]:
    """Modular inverses for 1..n (mod prime)."""
    inv = [0] * (n + 1)
    inv[1] = 1
    for i in range(2, n + 1):
        inv[i] = mod - (mod // i) * inv[mod % i] % mod
    return inv


# ------------------------ divisor sums from factorization ------------------------


def sigma_power_from_exps(exps: dict[int, int], s: int, mod: int) -> int:
    """Compute sigma_s(n) = sum_{d|n} d^s modulo mod, given prime exponents."""
    res = 1
    for p, e in exps.items():
        ps = pow(p, s, mod)
        if ps == 1:
            # geometric series degenerates
            term = (e + 1) % mod
        else:
            term = (pow(ps, e + 1, mod) - 1) % mod
            term = term * pow((ps - 1) % mod, mod - 2, mod) % mod
        res = (res * term) % mod
    return res


def n_mod_from_exps(exps: dict[int, int], mod: int) -> int:
    res = 1
    for p, e in exps.items():
        res = (res * pow(p, e, mod)) % mod
    return res


def build_sigma_data(exps: dict[int, int], mod: int) -> tuple[int, dict[int, int]]:
    """Return (n_mod, {1:sigma_1,3:sigma_3,...,11:sigma_11} mod mod)."""
    nmod = n_mod_from_exps(exps, mod)
    sig = {}
    for s in (1, 3, 5, 7, 9, 11):
        sig[s] = sigma_power_from_exps(exps, s, mod)
    return nmod, sig


# ----------------------------- Ramanujan tau for large n -----------------------------


def precompute_sigma1_upto(n: int) -> list[int]:
    sig1 = [0] * (n + 1)
    for d in range(1, n + 1):
        for m in range(d, n + 1, d):
            sig1[m] += d
    return sig1


def precompute_tau_upto(n: int, mod: int) -> list[int]:
    """Compute tau(1..n) modulo mod using D(Delta)=E2*Delta.

    Recurrence from E2*Delta = q d/dq Delta:
      (k-1) tau(k) = -24 * sum_{m=1}^{k-1} sigma1(m) * tau(k-m)
    """
    sig1 = precompute_sigma1_upto(n)
    inv = inv_table(n, mod)

    tau = [0] * (n + 1)
    tau[1] = 1

    # O(n^2) convolution; keep inner sums as small Python ints and mod once per k.
    for k in range(2, n + 1):
        total = 0
        # sum_{m=1}^{k-1} sig1[m] * tau[k-m]
        tm = tau
        sm = sig1
        for m in range(1, k):
            total += sm[m] * tm[k - m]
        total %= mod
        tau[k] = (-24 * total) % mod
        tau[k] = (tau[k] * inv[k - 1]) % mod
    return tau


def mat_mul(
    A: tuple[int, int, int, int], B: tuple[int, int, int, int], mod: int
) -> tuple[int, int, int, int]:
    a, b, c, d = A
    e, f, g, h = B
    return (
        (a * e + b * g) % mod,
        (a * f + b * h) % mod,
        (c * e + d * g) % mod,
        (c * f + d * h) % mod,
    )


def tau_prime_power(p: int, e: int, tau_p: int, mod: int) -> int:
    """Compute tau(p^e) modulo mod using tau(p^{k+1}) = tau(p)*tau(p^k) - p^{11}*tau(p^{k-1})."""
    if e == 0:
        return 1
    if e == 1:
        return tau_p % mod

    p11 = pow(p, 11, mod)
    # Companion matrix for the recurrence
    M = (tau_p % mod, (-p11) % mod, 1, 0)
    # Compute M^(e-1)
    R = (1, 0, 0, 1)
    exp = e - 1
    while exp:
        if exp & 1:
            R = mat_mul(R, M, mod)
        M = mat_mul(M, M, mod)
        exp >>= 1

    # Apply to vector [tau(p), tau(1)] = [tau_p, 1]
    a, b, c, d = R
    return (a * (tau_p % mod) + b) % mod


def tau_from_exps_factorial(exps: dict[int, int], tau_upto: list[int], mod: int) -> int:
    """Compute tau(n) mod mod for n given by prime exponents, using multiplicativity."""
    res = 1
    for p, e in exps.items():
        res = (res * tau_prime_power(p, e, tau_upto[p], mod)) % mod
    return res


# -------------------------- Eisenstein series coefficient helpers --------------------------

# Coefficients for normalized Eisenstein series E_k = 1 + c_k * sum_{n>=1} sigma_{k-1}(n) q^n
# (For k=2, it's quasimodular but the q-expansion still holds.)
INV_691 = pow(691, MOD - 2, MOD)
E_COEFF = {
    2: (-24) % MOD,
    4: 240 % MOD,
    6: (-504) % MOD,
    8: 480 % MOD,
    10: (-264) % MOD,
    12: (65520 * INV_691) % MOD,
}


def coeff_Ek(k: int, sig: dict[int, int]) -> int:
    """Coefficient of q^n in E_k for n>0, given sigma_{k-1}(n)."""
    if k == 2:
        return E_COEFF[2] * sig[1] % MOD
    return E_COEFF[k] * sig[k - 1] % MOD


def coeff_D_Ek(k: int, r: int, n_pows: list[int], sig: dict[int, int]) -> int:
    """Coefficient of q^n in D^r(E_k), n>0."""
    return n_pows[r] * coeff_Ek(k, sig) % MOD


# ----------------------------- E2 power coefficients -----------------------------

INV2 = (MOD + 1) // 2
INV5 = pow(5, MOD - 2, MOD)
INV7 = pow(7, MOD - 2, MOD)
INV24185 = pow(24185, MOD - 2, MOD)


def coeff_E2_pow(k: int, n_pows: list[int], sig: dict[int, int], tau_n: int) -> int:
    """Return coefficient of q^n in E2^k, for k<=6."""
    if k == 1:
        return coeff_Ek(2, sig)

    if k == 2:
        # E2^2 = E4 + 12 D(E2)
        return (coeff_Ek(4, sig) + 12 * coeff_D_Ek(2, 1, n_pows, sig)) % MOD

    if k == 3:
        # E2^3 = E6 + 9 D(E4) + 72 D^2(E2)
        return (
            coeff_Ek(6, sig)
            + 9 * coeff_D_Ek(4, 1, n_pows, sig)
            + 72 * coeff_D_Ek(2, 2, n_pows, sig)
        ) % MOD

    if k == 4:
        # E2^4 = E8 + 8 D(E6) + (216/5) D^2(E4) + 288 D^3(E2)
        c216_5 = 216 * INV5 % MOD
        return (
            coeff_Ek(8, sig)
            + 8 * coeff_D_Ek(6, 1, n_pows, sig)
            + c216_5 * coeff_D_Ek(4, 2, n_pows, sig)
            + 288 * coeff_D_Ek(2, 3, n_pows, sig)
        ) % MOD

    if k == 5:
        # E2^5 = E10 + (15/2) D(E8) + (240/7) D^2(E6) + 144 D^3(E4) + 864 D^4(E2)
        c15_2 = 15 * INV2 % MOD
        c240_7 = 240 * INV7 % MOD
        return (
            coeff_Ek(10, sig)
            + c15_2 * coeff_D_Ek(8, 1, n_pows, sig)
            + c240_7 * coeff_D_Ek(6, 2, n_pows, sig)
            + 144 * coeff_D_Ek(4, 3, n_pows, sig)
            + 864 * coeff_D_Ek(2, 4, n_pows, sig)
        ) % MOD

    if k == 6:
        # E2^6 = E12 - (4608/24185) Delta + (36/5) D(E10) + 30 D^2(E8)
        #        + (720/7) D^3(E6) + (2592/7) D^4(E4) + (10368/5) D^5(E2)
        c36_5 = 36 * INV5 % MOD
        c720_7 = 720 * INV7 % MOD
        c2592_7 = 2592 * INV7 % MOD
        c10368_5 = 10368 * INV5 % MOD
        cDelta = (-4608) % MOD * INV24185 % MOD
        return (
            coeff_Ek(12, sig)
            + cDelta * (tau_n % MOD)
            + c36_5 * coeff_D_Ek(10, 1, n_pows, sig)
            + 30 * coeff_D_Ek(8, 2, n_pows, sig)
            + c720_7 * coeff_D_Ek(6, 3, n_pows, sig)
            + c2592_7 * coeff_D_Ek(4, 4, n_pows, sig)
            + c10368_5 * coeff_D_Ek(2, 5, n_pows, sig)
        ) % MOD

    raise ValueError("k must be 1..6")


# ----------------------------- R_n computation -----------------------------


def comb_small(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    if k > n - k:
        k = n - k
    num = 1
    den = 1
    for i in range(1, k + 1):
        num *= n - (k - i)
        den *= i
    return num // den


def R_dim_at_n(dim: int, n_pows: list[int], sig: dict[int, int], tau_n: int) -> int:
    """Compute R_dim(M) mod MOD using the E2-power method (valid for dim<=6 here)."""
    inv12 = pow(12, MOD - 2, MOD)
    scale = pow(inv12, dim, MOD)

    s = 0
    for k in range(1, dim + 1):
        ck = comb_small(dim, k)
        term = ck * coeff_E2_pow(k, n_pows, sig, tau_n) % MOD
        if k % 2 == 1:
            s = (s - term) % MOD
        else:
            s = (s + term) % MOD

    return s * scale % MOD


def main() -> None:
    primes = sieve(10000)

    # Tests from the problem statement.
    ex10 = factorize_small(10, primes)
    nmod10, sig10 = build_sigma_data(ex10, MOD)
    n_pows10 = [1]
    for _ in range(5):
        n_pows10.append(n_pows10[-1] * nmod10 % MOD)
    assert R_dim_at_n(1, n_pows10, sig10, 0) == 36

    ex100 = factorize_small(100, primes)
    nmod100, sig100 = build_sigma_data(ex100, MOD)
    n_pows100 = [1]
    for _ in range(5):
        n_pows100.append(n_pows100[-1] * nmod100 % MOD)
    assert R_dim_at_n(2, n_pows100, sig100, 0) == 1873044

    # Precompute tau(p) mod MOD for primes p <= 10000 (needed for tau(10000!)).
    tau_upto = precompute_tau_upto(10000, MOD)

    ex100fac = factorial_prime_exponents(100, primes)
    nmod100fac, sig100fac = build_sigma_data(ex100fac, MOD)
    n_pows100fac = [1]
    for _ in range(5):
        n_pows100fac.append(n_pows100fac[-1] * nmod100fac % MOD)
    assert R_dim_at_n(2, n_pows100fac, sig100fac, 0) == 446575636

    # Main computation: R_6(10000!) mod MOD
    ex10000fac = factorial_prime_exponents(10000, primes)
    nmod, sig = build_sigma_data(ex10000fac, MOD)
    n_pows = [1]
    for _ in range(5):
        n_pows.append(n_pows[-1] * nmod % MOD)

    tau_fact = tau_from_exps_factorial(ex10000fac, tau_upto, MOD)

    ans = R_dim_at_n(6, n_pows, sig, tau_fact)
    print(ans)


if __name__ == "__main__":
    main()
