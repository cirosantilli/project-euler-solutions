#!/usr/bin/env python

ONES: dict[int, str] = {
    0: "",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}

TEENS: dict[int, str] = {
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
}

TENS: dict[int, str] = {
    2: "twenty",
    3: "thirty",
    4: "forty",
    5: "fifty",
    6: "sixty",
    7: "seventy",
    8: "eighty",
    9: "ninety",
}


def letters_1_to_99(n: int) -> int:
    """Letter count for 1..99."""
    assert 1 <= n <= 99
    if n < 10:
        return len(ONES[n])
    if 10 <= n <= 19:
        return len(TEENS[n])
    t, u = divmod(n, 10)
    return len(TENS[t]) + len(ONES[u])


def letters_in_number(n: int) -> int:
    """Letter count for 1..1000 using British 'and' rules. No spaces/hyphens."""
    assert 1 <= n <= 1000
    if n == 1000:
        return len("one") + len("thousand")  # "one thousand"
    if n < 100:
        return letters_1_to_99(n)

    h, r = divmod(n, 100)
    total = len(ONES[h]) + len("hundred")  # "<h> hundred"
    if r != 0:
        total += len("and")  # British usage
        if r < 100:
            total += letters_1_to_99(r)
    return total


def solve(n: int = 1000) -> int:
    """
    @formalsw: Requires(0 <= n <= 1000)
    @formalsw: Requires({0: "", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"} == ONES)
    @formalsw: Requires({10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen"} == TEENS)
    @formalsw: Requires({2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety"} == TENS)
    @formalsw: Py from project_euler_statements.p17 import naive
    @formalsw: Equiv(naive)
    """
    return sum(letters_in_number(i) for i in range(1, n + 1))


if __name__ == "__main__":
    # Given examples / checks
    assert sum(letters_in_number(i) for i in range(1, 6)) == 19
    assert letters_in_number(342) == 23
    assert letters_in_number(115) == 20

    print(solve())
