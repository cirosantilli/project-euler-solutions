#!/usr/bin/env python
from __future__ import annotations

def next_railway(pos: int) -> int:
    if pos < 5:
        return 5
    if pos < 15:
        return 15
    if pos < 25:
        return 25
    if pos < 35:
        return 35
    return 5


def next_utility(pos: int) -> int:
    if pos < 12:
        return 12
    if pos < 28:
        return 28
    return 12


def resolve_landing(square):
    """
    Uniform-card model: return integer weights with common denominator 256.
    This averages independent card draws, rather than conditioning on a cyclic deck.
    """
    if square == 30:
        return {(10, True): 256}
    if square in (2, 17, 33):
        return {(square, False): 224, (0, False): 16, (10, True): 16}
    if square in (7, 22, 36):
        weights = {(square, False): 96, (10, True): 16}
        for destination in (0, 11, 24, 39, 5, next_railway(square),
                            next_railway(square), next_utility(square), square - 3):
            if destination == 33:
                # Back three from Chance 36 draws Community Chest as well.
                for outcome, count in (((33, False), 14), ((0, False), 1), ((10, True), 1)):
                    weights[outcome] = weights.get(outcome, 0) + count
            else:
                outcome = destination, False
                weights[outcome] = weights.get(outcome, 0) + 16
        return weights
    return {(square, False): 256}


def encode_state(pos, doubles):
    return pos * 3 + doubles


def build_transition(dice_sides):
    if dice_sides < 1:
        raise ValueError("dice_sides must be positive")
    outcomes = []
    for move in range(2, 2 * dice_sides + 1):
        count = min(move - 1, 2 * dice_sides + 1 - move)
        double = 1 if move % 2 == 0 else 0
        if count > double:
            outcomes.append((move, False, count - double))
        if double:
            outcomes.append((move, True, 1))
    landings = [resolve_landing(pos) for pos in range(40)]
    transitions = []
    for pos in range(40):
        for doubles in range(3):
            row = {}
            for move, is_double, count in outcomes:
                if is_double and doubles == 2:
                    target = encode_state(10, 0)
                    row[target] = row.get(target, 0) + 256 * count
                else:
                    for (destination, sent), weight in landings[(pos + move) % 40].items():
                        next_doubles = 0 if sent or not is_double else doubles + 1
                        target = encode_state(destination, next_doubles)
                        row[target] = row.get(target, 0) + weight * count
            transitions.append(row)
    return transitions


def stationary_weights(transitions, denominator):
    """
    Solve the stationary equations exactly using fraction-free Bareiss elimination.
    Return integer weights proportional to the normalized stationary probabilities.
    """
    size = len(transitions)
    matrix = [[transitions[source].get(target, 0) - (denominator if source == target else 0)
               for source in range(size)] + [0] for target in range(size - 1)]
    matrix.append([1] * size + [1])
    previous = 1
    for column in range(size - 1):
        pivot_row = next((row for row in range(column, size) if matrix[row][column]), None)
        if pivot_row is None:
            raise ArithmeticError("stationary distribution is not unique")
        pivot_values = matrix[pivot_row]
        matrix[pivot_row] = matrix[column]
        matrix[column] = pivot_values
        pivot = pivot_values[column]
        for row in range(column + 1, size):
            factor = matrix[row][column]
            for j in range(column + 1, size + 1):
                matrix[row][j] = (pivot * matrix[row][j] - factor * pivot_values[j]) // previous
            matrix[row][column] = 0
        previous = pivot
    scale = matrix[-1][-2]
    if scale == 0:
        raise ArithmeticError("stationary distribution is not unique")
    weights = [0] * size
    for row in range(size - 1, -1, -1):
        weights[row] = (matrix[row][-1] * scale -
                        sum(matrix[row][j] * weights[j] for j in range(row + 1, size))) // matrix[row][row]
    if scale < 0:
        weights = [-weight for weight in weights]
    return weights


def solve(dice_sides: int) -> str:
    weights = stationary_weights(build_transition(dice_sides), 256 * dice_sides * dice_sides)
    square_weights = [sum(weights[pos * 3:pos * 3 + 3]) for pos in range(40)]
    return "".join(str(pos).zfill(2) for pos in sorted(range(40), key=lambda pos: (-square_weights[pos], pos))[:3])


if __name__ == "__main__":
    assert solve(1) == "101214"
    assert solve(6) == "102400"
    print(solve(4))
