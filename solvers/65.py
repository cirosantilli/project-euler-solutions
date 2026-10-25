#!/usr/bin/env python


def e_continued_fraction_coeffs(n: int) -> list[int]:
    """
    Returns the first n continued fraction coefficients for e:
    e = [2; 1, 2, 1, 1, 4, 1, 1, 6, 1, ...]
    """
    assert n >= 1
    coeffs = [2]
    for i in range(1, n):
        if i % 3 == 2:
            coeffs.append(2 * (i // 3 + 1))
        else:
            coeffs.append(1)
    return coeffs


def convergent_numerator_denominator(coeffs: list[int]) -> tuple[int, int]:
    """
    Given continued fraction coefficients [a0, a1, ..., a_{n-1}],
    compute the convergent p/q using standard recurrence:
      p[-2]=0, p[-1]=1
      q[-2]=1, q[-1]=0
      p[k]=a_k*p[k-1]+p[k-2], q[k]=a_k*q[k-1]+q[k-2]
    """
    p_nm2, p_nm1 = 0, 1
    q_nm2, q_nm1 = 1, 0

    for a in coeffs:
        p = a * p_nm1 + p_nm2
        q = a * q_nm1 + q_nm2
        p_nm2, p_nm1 = p_nm1, p
        q_nm2, q_nm1 = q_nm1, q

    return p_nm1, q_nm1


def digit_sum(x: int) -> int:
    return sum(int(c) for c in str(x))


def solve(n: int = 100) -> int:
    """
    Sum the numerator digits of the nth convergent of e.

    @formalsw: Requires(n >= 1)
    @formalsw: Py from project_euler_statements.p65 import naive
    @formalsw: Returns
    @formalsw: Equiv(naive)
    """
    older = 0
    numerator = 1
    for k in range(n):
        if k == 0:
            coefficient = 2
        elif k % 3 == 2:
            coefficient = 2 * (k // 3 + 1)
        else:
            coefficient = 1
        previous = numerator
        numerator = coefficient * numerator + older
        older = previous
    return digit_sum(numerator)


if __name__ == "__main__":
    # Test from the statement: 10th convergent numerator is 1457, digit sum is 17
    coeffs_10 = e_continued_fraction_coeffs(10)
    p10, q10 = convergent_numerator_denominator(coeffs_10)
    assert (p10, q10) == (1457, 536)
    assert digit_sum(p10) == 17

    print(solve())
