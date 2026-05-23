#!/usr/bin/env python

MOD = 1_000_000_007


def f(n: int, m: int, mod: int = MOD) -> int:
    layer: dict[tuple[int, int, int], int] = {(m, 0, 0): 1}
    total = 1
    explicit_limit = min(n, m + 2)

    for length in range(1, explicit_limit + 1):
        next_layer: dict[tuple[int, int, int], int] = {}

        for (remaining, lower, threshold), count in layer.items():
            upper = min(remaining + 1, length)
            for inversions in range(lower, upper):
                if inversions < threshold:
                    next_state = (
                        remaining - inversions,
                        inversions + 1,
                        threshold + 1,
                    )
                else:
                    next_state = (remaining - inversions, lower, inversions)

                next_layer[next_state] = (next_layer.get(next_state, 0) + count) % mod

        layer = next_layer
        total = (total + sum(layer.values())) % mod

    if n > m + 2:
        stable_count = sum(layer.values()) % mod
        total = (total + ((n - (m + 2)) % mod) * stable_count) % mod

    return total


def solve() -> None:
    assert f(2, 0) == 3
    assert f(4, 5) == 32
    assert f(10, 25) == 294_400

    print(f(10**18, 40))


if __name__ == "__main__":
    solve()
