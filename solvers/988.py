from typing import Dict, Tuple


def frog_sum(a: int, b: int) -> int:
    """
    Compute F(a, b): the sum of frog positions over all non-attacking configurations.

    A positive position x is allowed only when x is not representable as ua + vb with
    u, v >= 0. For coprime a, b, every such gap can be written uniquely as

        x = ab - ai - bj,

    with i, j >= 1 and ai + bj < ab.

    Two gaps correspond to attacking frogs exactly when their (i, j) pairs are
    comparable coordinatewise. So non-attacking configurations are antichains in a
    Ferrers-shaped poset, and those antichains are encoded by monotone column heights.
    """
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be positive")

    # The problem is symmetric in a and b, so keep the height small.
    if a > b:
        a, b = b, a

    if a == 1:
        return 0

    width = b - 1

    # h[i] = number of cells in column i of the Ferrers diagram.
    # Columns are 1-indexed to match the math.
    h = [0] * (width + 1)
    for i in range(1, width + 1):
        h[i] = (a * b - a * i - 1) // b

    # dp[height] = (count_of_prefixes, total_weight_of_finished_maximal_elements)
    # after processing columns up to the current one.
    dp: Dict[int, Tuple[int, int]] = {t: (1, 0) for t in range(h[1] + 1)}

    for i in range(2, width + 1):
        next_dp: Dict[int, Tuple[int, int]] = {}
        for prev_height, (count, total) in dp.items():
            limit = min(prev_height, h[i])
            for cur_height in range(limit + 1):
                add = 0
                if prev_height > cur_height and prev_height > 0:
                    add = a * b - a * (i - 1) - b * prev_height

                old_count, old_total = next_dp.get(cur_height, (0, 0))
                next_dp[cur_height] = (
                    old_count + count,
                    old_total + total + count * add,
                )
        dp = next_dp

    # Final sentinel column of height 0 closes the last maximal element, if any.
    answer = 0
    last_column = width
    for prev_height, (count, total) in dp.items():
        add = 0
        if prev_height > 0:
            add = a * b - a * last_column - b * prev_height
        answer += total + count * add

    return answer


def solve() -> None:
    # Tests given in the problem statement.
    assert frog_sum(3, 5) == 23
    assert frog_sum(5, 13) == 16336

    print(frog_sum(19, 53))


if __name__ == "__main__":
    solve()
