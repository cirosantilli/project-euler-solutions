from math import factorial


# Rank windows for the 10 possible straights.
# Ranks are indexed as:
# 0=A, 1=2, 2=3, ..., 8=9, 9=10, 10=J, 11=Q, 12=K
WINDOWS = [
    (0, 1, 2, 3, 4),
    (1, 2, 3, 4, 5),
    (2, 3, 4, 5, 6),
    (3, 4, 5, 6, 7),
    (4, 5, 6, 7, 8),
    (5, 6, 7, 8, 9),
    (6, 7, 8, 9, 10),
    (7, 8, 9, 10, 11),
    (8, 9, 10, 11, 12),
    (9, 10, 11, 12, 0),
]

# overlap[i][j] tells whether the i-th and j-th straight types share a rank.
OVERLAP = [[False] * 10 for _ in range(10)]
for i in range(10):
    set_i = set(WINDOWS[i])
    for j in range(10):
        OVERLAP[i][j] = bool(set_i & set(WINDOWS[j]))

# perms[n][k] = P(n, k) = n * (n-1) * ... * (n-k+1), for 0 <= n, k <= 4.
PERMS = [[0] * 5 for _ in range(5)]
for n in range(5):
    PERMS[n][0] = 1
    value = 1
    for k in range(1, 5):
        if k <= n:
            value *= n - (k - 1)
            PERMS[n][k] = value


def colorings_of_all_subsets(starts):
    """
    For the labeled straights in 'starts', compute for every subset mask the number
    of proper 4-colorings of the overlap graph induced by that subset.

    A color is a suit. A proper coloring means overlapping monochromatic straights
    must use different suits.
    """
    k = len(starts)
    full = 1 << k

    adjacency = [0] * k
    for i in range(k):
        si = starts[i]
        for j in range(i + 1, k):
            if OVERLAP[si][starts[j]]:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i

    independent = [False] * full
    independent[0] = True
    for mask in range(1, full):
        bit = mask & -mask
        vertex = bit.bit_length() - 1
        rest = mask ^ bit
        independent[mask] = independent[rest] and ((adjacency[vertex] & rest) == 0)

    # dp[mask] = number of ways to color 'mask' using the colors processed so far.
    dp = [0] * full
    dp[0] = 1
    for _ in range(4):
        new_dp = [0] * full
        for mask in range(full):
            total = 0
            sub = mask
            while True:
                if independent[sub]:
                    total += dp[mask ^ sub]
                if sub == 0:
                    break
                sub = (sub - 1) & mask
            new_dp[mask] = total
        dp = new_dp

    return dp


def labeled_count(starts):
    """
    Count disjoint choices for these labeled straight types, excluding straight flushes.

    'starts' is a list like [0, 0, 3, 7, ...] describing which of the 10 rank windows
    each labeled straight uses.
    """
    k = len(starts)
    full = 1 << k
    colorings = colorings_of_all_subsets(starts)

    total_active = [0] * 13
    active_masks_by_rank = [0] * 13
    for index, start in enumerate(starts):
        bit = 1 << index
        for rank in WINDOWS[start]:
            total_active[rank] += 1
            active_masks_by_rank[rank] |= bit

    total = 0
    for mask in range(full):
        ways = 1
        for rank in range(13):
            monochromatic_here = (active_masks_by_rank[rank] & mask).bit_count()
            flexible_here = total_active[rank] - monochromatic_here
            ways_at_rank = PERMS[4 - monochromatic_here][flexible_here]
            if ways_at_rank == 0:
                ways = 0
                break
            ways *= ways_at_rank

        term = colorings[mask] * ways
        if mask.bit_count() & 1:
            total -= term
        else:
            total += term

    return total


def feasible_type_counts(target):
    """
    Generate all multiplicity vectors (n_0, ..., n_9) for choosing 'target' straights,
    subject to no rank being needed more than four times.
    """
    counts = [0] * 10
    coverage = [0] * 13

    def backtrack(pos, remaining):
        if pos == 10:
            if remaining == 0:
                yield tuple(counts)
            return

        for amount in range(remaining + 1):
            ok = True
            for rank in WINDOWS[pos]:
                coverage[rank] += amount
                if coverage[rank] > 4:
                    ok = False
            counts[pos] = amount

            if ok:
                yield from backtrack(pos + 1, remaining - amount)

            for rank in WINDOWS[pos]:
                coverage[rank] -= amount

        counts[pos] = 0

    yield from backtrack(0, target)


def count_disjoint_straights(target):
    """
    Count unordered choices of 'target' disjoint straights from one deck.
    """
    total = 0
    for type_counts in feasible_type_counts(target):
        starts = []
        divisor = 1
        for start, amount in enumerate(type_counts):
            starts.extend([start] * amount)
            divisor *= factorial(amount)
        total += labeled_count(starts) // divisor
    return total


def main():
    assert count_disjoint_straights(1) == 10200
    assert count_disjoint_straights(2) == 31832952
    print(count_disjoint_straights(8))


if __name__ == "__main__":
    main()
