#!/usr/bin/env python


def factorial_digit_sum(n: int) -> int:
    """
    @formalsw: Py from project_euler_statements.p20 import naive
    @formalsw: Requires(n >= 0)
    @formalsw: Equiv(naive)
    """
    fact = 1
    for k in range(2, n + 1):
        fact *= k
    return sum(int(ch) for ch in str(fact))


def main() -> None:
    # Test case from the problem statement
    assert factorial_digit_sum(10) == 27

    print(factorial_digit_sum(100))


if __name__ == "__main__":
    main()
