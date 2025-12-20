#!/usr/bin/env python3
"""
Project Euler 773 - Ruff Numbers

We need F(97) mod 1_000_000_007.

No external libraries are used.
"""

MOD = 1_000_000_007


def first_k_primes_ending_in_7(k: int) -> list[int]:
    """
    Generate the first k primes that end in 7 (i.e., p % 10 == 7).

    Simple incremental primality test with trial division by previously found primes.
    k=97 is small, so this is easily fast enough.
    """
    if k <= 0:
        return []

    primes = [2]
    ending7 = []
    candidate = 3  # iterate over odd numbers only

    while len(ending7) < k:
        is_prime = True
        limit = int(candidate**0.5)
        for p in primes:
            if p > limit:
                break
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
            if candidate % 10 == 7:
                ending7.append(candidate)
        candidate += 2

    return ending7


def F(k: int, mod: int = MOD) -> int:
    """
    Compute F(k) modulo `mod` for Problem 773.

    Let M be the product of the first k primes ending in 7 (so N_k = 10*M).
    Then:
        F(k) ≡ M * ( A(k) + 5*phi(M) )  (mod mod)

    where phi(M) = Π(p-1) and
          A(k) = Σ_{s=0..k} (-1)^s * C(k,s) * q(s)
    with q(s) determined by s mod 4:
        s mod 4 : 0  1  2  3
        q(s)   : 7  1  3  9
    """
    primes7 = first_k_primes_ending_in_7(k)

    M_mod = 1
    phi_mod = 1
    for p in primes7:
        M_mod = (M_mod * p) % mod
        phi_mod = (phi_mod * (p - 1)) % mod

    # Compute A(k) using an O(k) binomial walk:
    # C(k,0)=1; C(k,s+1)=C(k,s)*(k-s)/(s+1)
    q_table = (7, 1, 3, 9)  # by s % 4
    A = 0
    c = 1  # C(k,0)

    for s in range(0, k + 1):
        term = (c * q_table[s & 3]) % mod
        if s & 1:
            A = (A - term) % mod
        else:
            A = (A + term) % mod

        if s < k:
            c = (c * (k - s)) % mod
            c = (c * pow(s + 1, mod - 2, mod)) % mod  # modular inverse (mod is prime)

    return (M_mod * ((A + 5 * phi_mod) % mod)) % mod


def main() -> None:
    # Test value from the problem statement
    assert F(3) == 76101452

    print(F(97))


if __name__ == "__main__":
    main()
