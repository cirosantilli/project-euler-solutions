#!/usr/bin/env python


def is_palindrome_base2(n: int) -> bool:
    s = bin(n)[2:]
    return s == s[::-1]


def generate_decimal_palindromes(limit: int) -> list[int]:
    pals = []
    prefix = 1
    while True:
        text = str(prefix)
        odd = int(text + text[-2::-1])
        if odd >= limit:
            return pals
        pals.append(odd)
        even = int(text + text[::-1])
        if even < limit:
            pals.append(even)
        prefix += 1


def solve(limit: int = 1_000_000) -> int:
    """
    @formalsw: Py from project_euler_statements.p36 import naive
    @formalsw: Requires(limit >= 0)
    @formalsw: Returns
    @formalsw: Equiv(naive)
    """
    total = 0
    for n in generate_decimal_palindromes(limit):
        # Binary palindrome cannot be even (would end with 0 -> leading zero when reversed)
        if (n & 1) == 0:
            continue
        if is_palindrome_base2(n):
            total += n
    return total


if __name__ == "__main__":
    # Provided example: 585 is palindromic in base 10 and base 2
    assert str(585) == str(585)[::-1]
    assert is_palindrome_base2(585)


    print(solve(1_000_000))
