#!/usr/bin/env python
"""
Project Euler 886: Coprime permutations.

For n = 2m, adding the vertex 1 turns the forced alternating path into a
Hamiltonian cycle in a balanced bipartite coprimality graph.  The count is then
evaluated by the determinant/permanent inclusion-exclusion identity described
in 886.md.
"""

from itertools import combinations
from math import gcd

MOD = 83_456_729


def binomial_table(n: int) -> list[list[int]]:
    comb = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        comb[i][0] = comb[i][i] = 1
        for j in range(1, i):
            comb[i][j] = comb[i - 1][j - 1] + comb[i - 1][j]
    return comb


def determinant_mod(matrix: list[list[int]], mod: int) -> int:
    size = len(matrix)
    if size == 0:
        return 1

    a = [row[:] for row in matrix]
    det = 1
    for col in range(size):
        pivot = -1
        for row in range(col, size):
            if a[row][col] % mod:
                pivot = row
                break
        if pivot < 0:
            return 0

        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
            det = -det

        pivot_value = a[col][col] % mod
        det = (det * pivot_value) % mod
        inv_pivot = pow(pivot_value, mod - 2, mod)

        for row in range(col + 1, size):
            if a[row][col]:
                factor = a[row][col] * inv_pivot % mod
                target = a[row]
                source = a[col]
                for j in range(col, size):
                    target[j] = (target[j] - factor * source[j]) % mod

    return det % mod


def grouped_classes(masks: list[int]) -> list[tuple[int, int, int]]:
    """
    Return (representative_index, count, mask), preserving first-seen order.
    """
    seen: dict[int, int] = {}
    classes: list[list[int]] = []
    for idx, mask in enumerate(masks):
        class_idx = seen.get(mask)
        if class_idx is None:
            seen[mask] = len(classes)
            classes.append([idx, 1, mask])
        else:
            classes[class_idx][1] += 1
    return [(rep, count, mask) for rep, count, mask in classes]


def P(n: int, mod: int = MOD) -> int:
    if n < 2:
        return 0
    if n & 1:
        raise ValueError("this implementation expects even n")

    m = n // 2
    odds = list(range(1, n, 2))
    evens = list(range(2, n + 1, 2))

    matrix = [[1 if gcd(u, v) == 1 else 0 for v in evens] for u in odds]

    row_masks = []
    for row in range(1, m):
        mask = 0
        for col in range(m):
            if matrix[row][col]:
                mask |= 1 << col
        row_masks.append(mask)

    col_masks = []
    for col in range(m):
        mask = 0
        for row in range(m):
            if matrix[row][col]:
                mask |= 1 << row
        col_masks.append(mask)

    row_classes = grouped_classes(row_masks)
    col_classes = grouped_classes(col_masks)
    row_class_count = len(row_classes)
    col_class_count = len(col_classes)

    # Edges between row-class representatives and column-class representatives.
    class_entry = [
        [matrix[row_rep + 1][col_rep] for col_rep, _, _ in col_classes]
        for row_rep, _, _ in row_classes
    ]

    # Row support over column classes; the distinguished row 1 sees every column.
    row_supports = []
    for row_rep, _, _ in row_classes:
        support = 0
        for col_class, (col_rep, _, _) in enumerate(col_classes):
            if matrix[row_rep + 1][col_rep]:
                support |= 1 << col_class
        row_supports.append(support)
    distinguished_support = (1 << col_class_count) - 1

    comb = binomial_table(m)
    permanent_cache: dict[tuple[int, int], int] = {}

    def grouped_permanent(row_selected: int, col_selected: int) -> int:
        key = (row_selected, col_selected)
        cached = permanent_cache.get(key)
        if cached is not None:
            return cached

        active_col_classes = []
        active_col_counts = []
        for col_class, (_, count, _) in enumerate(col_classes):
            remaining = count - ((col_selected >> col_class) & 1)
            if remaining:
                active_col_classes.append(col_class)
                active_col_counts.append(remaining)

        active_count = len(active_col_classes)
        row_groups: dict[int, int] = {((1 << active_count) - 1): 1}

        for row_class, (_, count, _) in enumerate(row_classes):
            remaining = count - ((row_selected >> row_class) & 1)
            if not remaining:
                continue

            support = 0
            original_support = row_supports[row_class]
            for new_col, old_col in enumerate(active_col_classes):
                if (original_support >> old_col) & 1:
                    support |= 1 << new_col
            row_groups[support] = row_groups.get(support, 0) + remaining

        supports = list(row_groups.keys())
        row_counts = list(row_groups.values())
        row_total = sum(row_counts)
        row_sums = [0] * len(supports)
        total = 0

        def rec(col_pos: int, chosen: int, coeff: int) -> None:
            nonlocal total
            if col_pos == active_count:
                product = coeff
                for value, count in zip(row_sums, row_counts):
                    if value == 0:
                        return
                    product = product * pow(value, count, mod) % mod
                if chosen & 1:
                    total -= product
                else:
                    total += product
                return

            class_size = active_col_counts[col_pos]
            affected = [
                idx
                for idx, support in enumerate(supports)
                if (support >> col_pos) & 1
            ]
            for take in range(class_size + 1):
                if take:
                    for idx in affected:
                        row_sums[idx] += take
                rec(
                    col_pos + 1,
                    chosen + take,
                    coeff * comb[class_size][take] % mod,
                )
                if take:
                    for idx in affected:
                        row_sums[idx] -= take

        rec(0, 0, 1)
        if row_total & 1:
            total = -total
        total %= mod
        permanent_cache[key] = total
        return total

    row_combinations: list[list[tuple[tuple[int, ...], int, int]]] = [
        [] for _ in range(col_class_count + 1)
    ]
    col_combinations: list[list[tuple[tuple[int, ...], int, int]]] = [
        [] for _ in range(col_class_count + 1)
    ]

    for size in range(col_class_count + 1):
        for chosen in combinations(range(row_class_count), size):
            mask = 0
            multiplicity = 1
            for row_class in chosen:
                mask |= 1 << row_class
                multiplicity *= row_classes[row_class][1]
            row_combinations[size].append((chosen, mask, multiplicity % mod))

        for chosen in combinations(range(col_class_count), size):
            mask = 0
            multiplicity = 1
            for col_class in chosen:
                mask |= 1 << col_class
                multiplicity *= col_classes[col_class][1]
            col_combinations[size].append((chosen, mask, multiplicity % mod))

    answer = 0
    for size in range(col_class_count + 1):
        sign = -1 if size & 1 else 1
        for row_choice, row_mask, row_mult in row_combinations[size]:
            for col_choice, col_mask, col_mult in col_combinations[size]:
                det_matrix = [
                    [class_entry[row_class][col_class] for col_class in col_choice]
                    for row_class in row_choice
                ]
                det = determinant_mod(det_matrix, mod)
                if det == 0:
                    continue

                perm = grouped_permanent(row_mask, col_mask)
                term = row_mult * col_mult % mod
                term = term * det * det % mod
                term = term * perm * perm % mod
                answer += sign * term

    return answer % mod


def main() -> None:
    assert P(4, MOD) == 2
    assert P(10, MOD) == 576
    print(P(34, MOD))


if __name__ == "__main__":
    main()
