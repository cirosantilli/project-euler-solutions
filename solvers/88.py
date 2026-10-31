#!/usr/bin/env python

from __future__ import annotations


def solve(k_max: int) -> int:
    """
    Return the sum of distinct minimal product-sum numbers for 2 <= k <= k_max.
    Uses the standard factor-combination DFS:
      For a factor multiset (>=2 factors only), with product P, sum S, length t,
      we can append (P - S) ones to make sum equal product.
      Then the effective k is: k = t + (P - S).

    @formalsw: Returns
    """
    if k_max < 2:
        return 0
    limit = 2 * k_max  # Factors 2 and k, with k-2 ones, give product and sum 2*k.
    sentinel = limit + 1
    best: list[int] = [sentinel] * (k_max + 1)

    def dfs(start_factor: int, prod: int, summ: int, terms: int) -> None:
        # try adding next factor f >= start_factor to keep nondecreasing order
        max_f = limit // prod
        for f in range(start_factor, max_f + 1):
            new_prod = prod * f
            new_sum = summ + f
            new_terms = terms + 1

            k = new_terms + (new_prod - new_sum)
            if k <= k_max:
                if new_prod < best[k]:
                    best[k] = new_prod
                dfs(f, new_prod, new_sum, new_terms)

    dfs(2, 1, 0, 0)

    uniq: set[int] = set(best[2:])  # ignore k=0,1
    return sum(uniq)


if __name__ == "__main__":
    assert solve(6) == 30
    assert solve(12) == 61
    print(solve(12000))
