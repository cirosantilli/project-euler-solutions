#!/usr/bin/env python


def precompute_factorials(base: int = 10) -> list[int]:
    facts = [1]
    f = 1
    for digit in range(1, base):
        f *= digit
        facts.append(f)
    return facts


def digit_factorial_sum(n: int, facts: list[int], base: int = 10) -> int:
    total = 0
    while n:
        n, remainder = divmod(n, base)
        total += facts[remainder]
    return total


def solve(base: int = 10) -> int:
    """
    Sum the numbers with at least two base-b digits equal to their digit-factorial sum.

    @formalsw: Requires(base >= 2)
    @formalsw: Py from project_euler_statements.p34 import naive
    @formalsw: Returns
    @formalsw: Equiv(naive)
    """
    facts = precompute_factorials(base)
    max_fact = facts[base - 1]
    digits = 1
    while digits * max_fact >= base ** (digits - 1):
        digits += 1
    upper = (digits - 1) * max_fact

    total = 0
    for n in range(base, upper + 1):
        if n == digit_factorial_sum(n, facts, base):
            total += n
    return total


if __name__ == "__main__":
    print(solve())
