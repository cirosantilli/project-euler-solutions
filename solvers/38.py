#!/usr/bin/env python


def concatenated_product(x: int, width: int = 9) -> tuple[str, int]:
    parts = []
    n = 0
    total_len = 0
    while total_len < width:
        n += 1
        s = str(x * n)
        parts.append(s)
        total_len += len(s)
    return "".join(parts), n


def is_pandigital(s: str, width: int = 9) -> bool:
    if len(s) != width:
        return False
    mask = 0
    for ch in s:
        digit = ord(ch) - 48
        if digit < 1:
            return False
        if digit > width:
            return False
        bit = 1 << digit
        if mask & bit:
            return False
        mask |= bit
    return True


def solve(width: int = 9) -> int:
    """
    @formalsw: Py from project_euler_statements.p38 import naive
    @formalsw: Returns
    @formalsw: Equiv(naive)
    """
    if width < 2:
        return 0
    if width > 9:
        return 0
    best = 0
    for x in range(1, 10 ** (width // 2)):
        s, n = concatenated_product(x, width)
        if n > 1 and is_pandigital(s, width):
            val = int(s)
            if val > best:
                best = val
    return best


def _concat_fixed(x: int, n: int) -> str:
    return "".join(str(x * k) for k in range(1, n + 1))


if __name__ == "__main__":
    assert _concat_fixed(192, 3) == "192384576"
    assert is_pandigital("192384576")
    assert _concat_fixed(9, 5) == "918273645"
    assert is_pandigital("918273645")
    print(solve())
