#!/usr/bin/env python3
"""
Project Euler 739
-----------------
No external libraries are used.

We compute f(n) modulo 1_000_000_007.

The derivation (see README.md) yields an algebraic generating function whose
coefficients satisfy a 4th-order linear recurrence with linear coefficients.
We evaluate that recurrence up to n = 10**8 using blockwise modular inverses
(to avoid doing an expensive modular exponentiation per step).
"""

MOD = 1_000_000_007
INV2 = (MOD + 1) // 2  # inverse of 2 modulo MOD


def _inverses_consecutive(start: int, length: int) -> list[int]:
    """
    Return modular inverses of 2*start, 2*(start+1), ..., 2*(start+length-1) (mod MOD),
    using one modular exponentiation and O(length) multiplications.

    (Equivalently: INV2 * inv(start+i) for each i.)

    Preconditions: 1 <= start and start+length-1 < MOD (so none are 0 mod MOD).
    """
    # prefix products
    pref = [1] * (length + 1)
    x = start
    for i in range(length):
        pref[i + 1] = (pref[i] * x) % MOD
        x += 1

    inv_total = pow(pref[length], MOD - 2, MOD)

    invs = [0] * length
    x = start + length - 1
    for i in range(length - 1, -1, -1):
        invs[i] = ((inv_total * pref[i]) % MOD) * INV2 % MOD
        inv_total = (inv_total * x) % MOD
        x -= 1
    return invs


def f(n: int) -> int:
    """
    Compute f(n) mod MOD, where f is defined in the problem.

    We work with the generating function coefficient a_m = [x^m] F(x),
    and f(n) = a_{n-1}.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if n == 1:
        return 1

    m = n - 1  # coefficient index
    # Initial terms (a_0..a_3)
    if m == 0:
        return 1
    if m == 1:
        return 3
    if m == 2:
        return 7
    if m == 3:
        return 21

    # State holds a_k, a_{k+1}, a_{k+2}, a_{k+3}
    a0, a1, a2, a3 = 1, 3, 7, 21

    # Recurrence for k>=0:
    # (4k+2)a_k + (23k+26)a_{k+1} + (22k+62)a_{k+2} - (15k+50)a_{k+3} + (2k+8)a_{k+4} = 0
    # a_{k+4} = ((15k+50)a_{k+3} - (4k+2)a_k - (23k+26)a_{k+1} - (22k+62)a_{k+2}) / (2k+8)
    #
    # We use inv(2k+8) = INV2 * inv(k+4).

    steps = m - 3  # after this many steps, a3 == a_m
    # Linear coefficients at k=0
    c0, c1, c2, c3 = 2, 26, 62, 50  # 4k+2, 23k+26, 22k+62, 15k+50
    denom = 4  # k+4

    # Tuneable: trade memory vs number of pow() calls
    BLOCK = 200_000

    remaining = steps
    while remaining:
        L = BLOCK if remaining > BLOCK else remaining
        invs = _inverses_consecutive(denom, L)

        # Tight loop
        for j in range(L):
            inv_d = invs[j]

            t = (c3 * a3 - c0 * a0 - c1 * a1 - c2 * a2) % MOD
            a4 = (t * inv_d) % MOD

            a0 = a1
            a1 = a2
            a2 = a3
            a3 = a4

            c0 += 4
            c1 += 23
            c2 += 22
            c3 += 15
            denom += 1

        remaining -= L

    return a3


def main() -> None:
    # Test values from the problem statement
    assert f(8) == 2663
    assert f(20) == 742296999

    target = 10**8
    print(f(target) % MOD)


if __name__ == "__main__":
    main()
