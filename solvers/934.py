#!/usr/bin/env python3
"""Project Euler 934: Unlucky Primes

We define u(n) as the smallest prime p such that (n mod p) is NOT a multiple of 7.
Let U(N) = sum_{n=1..N} u(n).

This program computes U(10^17) without external libraries.
"""

from __future__ import annotations


def prime_generator():
    """Yield primes in increasing order (simple incremental trial division)."""
    yield 2
    primes = [2]
    x = 3
    while True:
        r = int(x**0.5)
        is_p = True
        for p in primes:
            if p > r:
                break
            if x % p == 0:
                is_p = False
                break
        if is_p:
            primes.append(x)
            yield x
        x += 2


def inv_mod(a: int, m: int) -> int:
    """Modular inverse of a modulo m (a and m coprime)."""
    a %= m
    # Extended Euclid
    t0, t1 = 0, 1
    r0, r1 = m, a
    while r1:
        q = r0 // r1
        t0, t1 = t1, t0 - q * t1
        r0, r1 = r1, r0 - q * r1
    if r0 != 1:
        raise ValueError("inverse does not exist")
    return t0 % m


def u(n: int) -> int:
    """Compute u(n) directly (only used for small test asserts)."""
    for p in prime_generator():
        if (n % p) % 7 != 0:
            return p


def U(N: int) -> int:
    """Compute U(N) using CRT residue construction + a switch to explicit enumeration."""
    ans = 0

    # c_prev = count of numbers in [1..N] that satisfy the constraints for primes processed so far.
    # Initially, with no primes processed, every n in [1..N] satisfies: c_prev = N.
    c_prev = N

    # Phase 1: maintain the set of valid residues modulo M (product of processed primes)
    # while M stays <= N. Residues are stored as integers in [0, M-1].
    M = 1
    residues = [0]

    # Phase 2: once M exceeds N, each valid residue corresponds to at most one n in [1..N].
    # We then keep the explicit list of surviving n values (positive, <= N).
    survivors = None  # type: list[int] | None

    for p in prime_generator():
        # Allowed residues modulo p are exactly the multiples of 7 in [0, p-1].
        allowed = range(0, p, 7)

        if survivors is None:
            # Still in residue/CRT mode.
            M_new = M * p
            inv = inv_mod(M, p)

            if M_new <= N:
                M_old = M
                new_residues = []
                append = new_residues.append
                for r in residues:
                    for s in allowed:
                        t = ((s - r) * inv) % p
                        append(r + M_old * t)

                M = M_new
                residues = new_residues

                # Count numbers <= N matching these residues (period M).
                q, rem = divmod(N, M)
                extra = 0
                for r in residues:
                    if 0 < r <= rem:
                        extra += 1
                c = q * len(residues) + extra

            else:
                # Switch: build the explicit surviving numbers <= N.
                M_old = M
                survivors = []
                append = survivors.append
                for r in residues:
                    for s in allowed:
                        t = ((s - r) * inv) % p
                        x = r + M_old * t  # x is the unique solution modulo M_new
                        if 0 < x <= N:
                            append(x)
                M = M_new
                c = len(survivors)

        else:
            # Explicit enumeration mode: just filter survivors.
            # Keep n if (n mod p) is divisible by 7.
            survivors = [n for n in survivors if (n % p) % 7 == 0]
            c = len(survivors)

        # Numbers that stop being valid at prime p have u(n) = p.
        ans += p * (c_prev - c)
        c_prev = c

        if c == 0:
            break

    return ans


def main() -> None:
    # Asserts from the problem statement
    assert u(14) == 3
    assert u(147) == 2
    assert u(1470) == 13
    assert U(1470) == 4293

    print(U(10**17))


if __name__ == "__main__":
    main()
