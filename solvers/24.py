#!/usr/bin/env python


def valid_permutation_input(digits: list[int], n: int) -> bool:
    """
    Check the strictly increasing input order and the one-based rank bounds.
    """
    for i in range(1, len(digits)):
        if digits[i - 1] >= digits[i]:
            return False
    count = 1
    for i in range(2, len(digits) + 1):
        count *= i
    return 1 <= n <= count


def nth_lexicographic_permutation(digits: list[int], n: int) -> str:
    """
    Returns the n-th lexicographic permutation (1-indexed) of the given digits.

    @formalsw: Py from project_euler_statements.p24 import naive_text as naive
    @formalsw: Requires(valid_permutation_input(digits, n))
    @formalsw: Equiv(naive)
    """
    k = len(digits)

    # Precompute factorials up to k
    fact = [1] * (k + 1)
    for i in range(2, k + 1):
        fact[i] = fact[i - 1] * i

    # Convert to 0-indexed rank
    n -= 1

    available = digits[:]
    out: list[str] = []
    for i in range(k, 0, -1):
        block = fact[i - 1]
        idx = n // block
        n %= block
        out.append(str(available.pop(idx)))
    return "".join(out)


def main() -> None:
    digits = list(range(10))
    ans = nth_lexicographic_permutation(digits, 1_000_000)
    print(ans)


if __name__ == "__main__":
    main()
