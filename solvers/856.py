#!/usr/bin/env python

from functools import lru_cache


def prob_pair_in_first_two() -> float:
    return 3.0 / 51.0


@lru_cache(maxsize=None)
def expected_additional(
    current: int, a0: int, a1: int, a2: int, a3: int, a4: int
) -> float:
    buckets = (a0, a1, a2, a3, a4)
    remaining = current + a1 + 2 * a2 + 3 * a3 + 4 * a4
    if remaining == 0:
        return 0.0

    result = 1.0
    for count in range(1, 5):
        ranks = buckets[count]
        if ranks == 0:
            continue

        next_buckets = list(buckets)
        next_buckets[count] -= 1
        next_buckets[current] += 1
        result += (
            count * ranks / remaining * expected_additional(count - 1, *next_buckets)
        )

    return result


def expected_cards_drawn() -> float:
    return 1.0 + expected_additional(3, 0, 0, 0, 0, 12)


def main() -> None:
    assert abs(prob_pair_in_first_two() - 1.0 / 17.0) < 1e-15

    answer = expected_cards_drawn()
    assert 2.0 <= answer <= 52.0
    print(f"{answer:.8f}")


if __name__ == "__main__":
    main()
