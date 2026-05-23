#!/usr/bin/env python
"""
Project Euler 810: XOR-Primes

Interpret each integer as a polynomial over F_2.  XOR-primes are the codes of
irreducible polynomials, so we can sieve odd polynomial products.  Consecutive
monic cofactors are generated in Gray-code order, making each next product one
XOR with a shifted copy of the base polynomial.
"""

TARGET = 5_000_000


def mobius(n: int) -> int:
    result = 1
    p = 2
    while p * p <= n:
        if n % p == 0:
            n //= p
            if n % p == 0:
                return 0
            result = -result
        p += 1 if p == 2 else 2
    if n > 1:
        result = -result
    return result


def irreducible_count(degree: int) -> int:
    total = 0
    for d in range(1, degree + 1):
        if degree % d == 0:
            total += mobius(d) * (1 << (degree // d))
    return total // degree


def search_bit_limit(rank: int) -> int:
    count = 0
    degree = 0
    while count < rank:
        degree += 1
        count += irreducible_count(degree)
    return degree + 1


def xor_product(a: int, b: int) -> int:
    product = 0
    while b:
        if b & 1:
            product ^= a
        a <<= 1
        b >>= 1
    return product


def nth_xor_prime(rank: int) -> int:
    if rank == 1:
        return 2

    bit_limit = search_bit_limit(rank)
    limit = 1 << bit_limit

    # Index k represents the odd code 2*k+1.  Code 1 is not a prime.
    composite = bytearray(limit >> 1)
    composite[0] = 1

    found = 1  # the even irreducible polynomial x, encoded by 2
    mark = composite

    for base in range(3, limit, 2):
        if mark[base >> 1]:
            continue

        found += 1
        if found == rank:
            return base

        degree = base.bit_length() - 1
        max_cofactor_degree = bit_limit - degree - 1

        for cofactor_degree in range(degree, max_cofactor_degree + 1):
            # Odd monic cofactor x^j + 1, then all variants whose internal
            # coefficients are traversed in Gray-code order.
            product = (base << cofactor_degree) ^ base
            mark[product >> 1] = 1

            variants = 1 << (cofactor_degree - 1)
            for n in range(1, variants):
                toggled_bit = (n & -n).bit_length()
                product ^= base << toggled_bit
                mark[product >> 1] = 1

    raise RuntimeError("search range was too small")


def first_xor_primes(count: int) -> list[int]:
    primes: list[int] = []
    x = 2
    while len(primes) < count:
        is_prime = True
        for a in range(2, x):
            for b in range(2, x):
                if xor_product(a, b) == x:
                    is_prime = False
                    break
            if not is_prime:
                break
        if is_prime:
            primes.append(x)
        x += 1
    return primes


def main() -> None:
    assert xor_product(7, 3) == 9
    assert xor_product(3, 3) == 5
    assert first_xor_primes(10) == [2, 3, 7, 11, 13, 19, 25, 31, 37, 41]
    assert nth_xor_prime(10) == 41
    print(nth_xor_prime(TARGET))


if __name__ == "__main__":
    main()
