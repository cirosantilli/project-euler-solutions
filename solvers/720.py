from __future__ import annotations

from array import array
from typing import Tuple

MOD = 1_000_000_007


def build_arrays(target_size: int) -> Tuple[array, array]:
    if target_size < 4 or target_size & (target_size - 1):
        raise ValueError("target_size must be a power of two >= 4")
    vals = array("I", [1, 3, 2, 4])
    codes = array("I", [0, 1, 0, 0])
    size = 4
    while size < target_size:
        m = size
        n = m << 1
        new_vals = array("I", [0]) * n
        new_codes = array("I", [0]) * n
        for i in range(m - 1):
            v = vals[i]
            new_vals[i] = (v << 1) - 1
            new_codes[i] = (v - 1) + codes[i]
        new_vals[m - 1] = 2
        new_codes[m - 1] = 0
        v_last = vals[m - 1]
        new_vals[m] = (v_last << 1) - 1
        new_codes[m] = m - 2
        for j in range(1, m):
            v = vals[j]
            new_vals[m + j] = v << 1
            new_codes[m + j] = codes[j]
        vals, codes = new_vals, new_codes
        size = n
    return vals, codes


def rank_from_prev(vals: array, codes: array, mod: int = MOD) -> int:
    m = len(vals)
    rank = 0
    fact = 1
    step = 1
    codes_local = codes
    vals_local = vals
    for j in range(m - 1, 0, -1):
        l = codes_local[j]
        rank = (rank + l * fact) % mod
        fact = (fact * step) % mod
        step += 1
    l = m - 2
    rank = (rank + l * fact) % mod
    fact = (fact * step) % mod
    step += 1
    fact = (fact * step) % mod
    step += 1
    for i in range(m - 2, -1, -1):
        v = vals_local[i]
        l = (v - 1) + codes_local[i]
        rank = (rank + l * fact) % mod
        fact = (fact * step) % mod
        step += 1
    return rank


def solve_power(k: int) -> int:
    if k == 0:
        return 1
    if k == 1:
        return 1
    if k == 2:
        return 3
    vals, codes = build_arrays(1 << (k - 1))
    rank0 = rank_from_prev(vals, codes, MOD)
    return (rank0 + 1) % MOD


def main() -> None:
    assert solve_power(2) == 3
    assert solve_power(3) == 2295
    assert solve_power(5) == 641839205
    ans = solve_power(25)
    print(ans % MOD)


if __name__ == "__main__":
    main()
