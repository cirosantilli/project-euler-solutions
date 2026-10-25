#!/usr/bin/env python


def solve(n: int, k: int = 10) -> str:
    """
    @formalsw: Requires(k >= 0)
    @formalsw: Py from project_euler_statements.p48 import naive_digits
    @formalsw: Equiv(naive_digits)
    """
    mod = 10**k
    s = 0
    for i in range(1, n + 1):
        s = (s + pow(i, i, mod)) % mod
    return str(s).zfill(k)


def exact_self_powers_sum(n: int) -> int:
    return sum(pow(i, i) for i in range(1, n + 1))


if __name__ == "__main__":
    # Test case from the problem statement:
    # 1^1 + 2^2 + ... + 10^10 = 10405071317
    assert exact_self_powers_sum(10) == 10405071317
    assert solve(10, 10) == "0405071317"

    # Required result:
    print(solve(1000, 10))
