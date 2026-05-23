#!/usr/bin/env python
"""
Project Euler 376: Nontransitive Sets of Dice

Dynamic programming over face values.  A state stores how many faces have been
assigned to each die and the three directed win counts, capped at 19.
"""

THRESHOLD = 19
WIN_STATES = 20
WIN_SPACE = WIN_STATES**3


def pack_state(a: int, b: int, c: int, ba: int, cb: int, ac: int) -> int:
    abc = (a * 7 + b) * 7 + c
    return ((abc * WIN_STATES + ba) * WIN_STATES + cb) * WIN_STATES + ac


def build_transitions() -> list[list[tuple[int, int, int, int, int, int, int]]]:
    transitions: list[list[tuple[int, int, int, int, int, int, int]]] = [
        [] for _ in range(7 * 7 * 7)
    ]

    for a_used in range(7):
        for b_used in range(7):
            for c_used in range(7):
                abc = (a_used * 7 + b_used) * 7 + c_used
                options = transitions[abc]

                for da in range(7 - a_used):
                    for db in range(7 - b_used):
                        for dc in range(7 - c_used):
                            na = a_used + da
                            nb = b_used + db
                            nc = c_used + dc
                            next_base = ((na * 7 + nb) * 7 + nc) * WIN_SPACE

                            # New faces at this value beat only earlier lower faces.
                            d_ba = db * a_used
                            d_cb = dc * b_used
                            d_ac = da * c_used

                            options.append((next_base, na, nb, nc, d_ba, d_cb, d_ac))

    return transitions


TRANSITIONS = build_transitions()
GOAL = pack_state(6, 6, 6, 19, 19, 19)


def count_nontransitive_sets(n: int) -> int:
    current: dict[int, int] = {0: 1}

    for value in range(1, n + 1):
        next_layer: dict[int, int] = {}
        is_last_value = value == n

        for state, ways in current.items():
            ac = state % WIN_STATES
            tmp = state // WIN_STATES
            cb = tmp % WIN_STATES
            tmp //= WIN_STATES
            ba = tmp % WIN_STATES
            abc = tmp // WIN_STATES

            for next_base, na, nb, nc, d_ba, d_cb, d_ac in TRANSITIONS[abc]:
                nba = ba + d_ba
                if nba > 19:
                    nba = 19
                ncb = cb + d_cb
                if ncb > 19:
                    ncb = 19
                nac = ac + d_ac
                if nac > 19:
                    nac = 19

                # Even best-case future faces cannot rescue this state.
                if nba + 6 * (6 - nb) < THRESHOLD:
                    continue
                if ncb + 6 * (6 - nc) < THRESHOLD:
                    continue
                if nac + 6 * (6 - na) < THRESHOLD:
                    continue

                if is_last_value and (na != 6 or nb != 6 or nc != 6):
                    continue

                key = next_base + nba * 400 + ncb * 20 + nac
                next_layer[key] = next_layer.get(key, 0) + ways

        current = next_layer

    return current.get(GOAL, 0) // 3


def solve() -> int:
    assert count_nontransitive_sets(7) == 9780
    return count_nontransitive_sets(30)


if __name__ == "__main__":
    print(solve())
