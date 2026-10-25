#!/usr/bin/env python

def digit_signature(n: int) -> tuple[int, ...]:
    """
    Return the ten decimal digit counts; nonpositive inputs have zero counts.

    @formalsw: Returns
    """
    cnt = [0] * 10
    while n > 0:
        cnt[n % 10] += 1
        n //= 10
    return tuple(cnt)


def solve(limit_mul: int = 6) -> int:
    """
    @formalsw: Requires(limit_mul >= 2)
    @formalsw: Requires(limit_mul <= 6)
    @formalsw: Py from project_euler_statements.p52 import naive
    @formalsw: Equiv(naive)
    """
    d = 1
    while True:
        lo = 10 ** (d - 1)
        hi = (10**d - 1) // limit_mul  # ensure limit_mul*x still has <= d digits
        for x in range(lo, hi + 1):
            sig = digit_signature(x)
            ok = True
            for m in range(2, limit_mul + 1):
                if digit_signature(m * x) != sig:
                    ok = False
                    break
            if ok:
                return x
        d += 1


if __name__ == "__main__":
    print(solve(6))
