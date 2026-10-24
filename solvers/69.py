#!/usr/bin/env python


def solve(limit: int) -> int:
    """
    Return the least positive maximizer of n/phi(n) up to the inclusive bound.
    Return one for bounds below one.

    @formalsw: Py from project_euler_statements.p69 import naive
    @formalsw: Returns
    @formalsw: Equiv(naive)
    """
    if limit < 2:
        return 1
    product = 2
    primes: list[int] = [2]
    candidate = 3
    while product * candidate <= limit:
        is_prime = True
        for divisor in primes:
            if divisor * divisor > candidate:
                break
            if candidate % divisor == 0:
                is_prime = False
                break
        if is_prime:
            product *= candidate
            primes.append(candidate)
        candidate += 2
    return product


if __name__ == "__main__":
    assert solve(10) == 6
    print(solve(1_000_000))
