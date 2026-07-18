#!/usr/bin/env python
from __future__ import annotations

MODULUS = 1_000_000_009
TARGET = 10_000_000

# A creative-telescoping recurrence for the algebraic generating function
# described in 1007.md.  COEFFICIENTS[j] stores p_j(n), low degree first, in
# sum p_j(n) A(n + j) = 0.
COEFFICIENTS = [
    [
        564542217,
        325536195,
        625565095,
        258562828,
        50309146,
        721049696,
        308910873,
        234609857,
        352444580,
        112684852,
        526098889,
        775427372,
    ],
    [
        756354407,
        876712396,
        925232674,
        209529903,
        431043426,
        98036193,
        411159730,
        650080946,
        586735744,
        802393957,
        917731008,
        662122179,
    ],
    [
        770353449,
        88154905,
        504692236,
        655625322,
        572603598,
        991869888,
        915468074,
        693094161,
        179919603,
        873974932,
        986423939,
        193800523,
    ],
    [
        547044626,
        431921333,
        336871141,
        405939744,
        684304146,
        406844875,
        110268327,
        6418661,
        533987403,
        378546146,
        256517355,
        458272153,
    ],
    [
        584021986,
        820086388,
        399407432,
        363471199,
        28065946,
        158462463,
        212035288,
        642246323,
        128937736,
        729528060,
        400553445,
        904647134,
    ],
    [
        130412455,
        271277001,
        847789671,
        369809900,
        273501501,
        415528861,
        103000357,
        79415045,
        892620950,
        963713973,
        920486784,
        150788293,
    ],
    [
        841263954,
        877714966,
        231261912,
        145551205,
        358810941,
        739118822,
        741630308,
        51109887,
        572039589,
        586686863,
        780306805,
        753840921,
    ],
    [
        924453715,
        724710183,
        251343960,
        617970784,
        108843307,
        820712895,
        926891587,
        544054172,
        772790579,
        265582300,
        658210597,
        233845863,
    ],
    [
        348850580,
        18190700,
        135200071,
        740796395,
        394618090,
        119264792,
        469650557,
        922517855,
        620146658,
        577934689,
        143405744,
        743846391,
    ],
    [
        363458011,
        696243382,
        636220249,
        229509005,
        303332388,
        631146571,
        677124719,
        152104366,
        123244817,
        586518959,
        314105934,
        924615366,
    ],
    [
        193410267,
        911570216,
        570155372,
        848170664,
        434343571,
        332813236,
        991880938,
        357506900,
        867920836,
        464991138,
        0,
        1,
    ],
]

INITIAL_VALUES = [
    0,
    MODULUS - 1,
    MODULUS - 2,
    MODULUS - 6,
    MODULUS - 20,
    MODULUS - 76,
    MODULUS - 314,
    MODULUS - 1409,
    MODULUS - 6732,
    MODULUS - 33900,
    MODULUS - 177666,
]


def polynomial_value(coefficients: list[int], n: int) -> int:
    value = 0
    for coefficient in reversed(coefficients):
        value = (value * n + coefficient) % MODULUS
    return value


def alternating_difference_sum(limit: int) -> int:
    values = INITIAL_VALUES[:]
    if limit < len(values):
        return values[limit]

    for n in range(1, limit - 9):
        total = 0
        for offset in range(10):
            total += polynomial_value(COEFFICIENTS[offset], n) * values[n + offset]
        denominator = polynomial_value(COEFFICIENTS[10], n)
        values.append((-total * pow(denominator, MODULUS - 2, MODULUS)) % MODULUS)
    return values[limit]


def main() -> None:
    assert alternating_difference_sum(3) == MODULUS - 6
    assert alternating_difference_sum(10) == MODULUS - 177666
    assert alternating_difference_sum(100) == 71792794
    print(alternating_difference_sum(TARGET))


if __name__ == "__main__":
    main()
