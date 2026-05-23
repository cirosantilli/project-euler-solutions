#!/usr/bin/env python

MODULO = 1_000_000_000


def solve(num_digits: int) -> int:
    max_sum = num_digits * 9 * 9
    counts = [0] * (max_sum + 1)
    values = [0] * (max_sum + 1)
    counts[0] = 1

    place_value = 1
    for length in range(1, num_digits + 1):
        next_counts = [0] * (max_sum + 1)
        next_values = [0] * (max_sum + 1)
        previous_limit = 81 * (length - 1)

        for square_sum in range(previous_limit + 1):
            count = counts[square_sum]
            value_sum = values[square_sum]
            if count == 0 and value_sum == 0:
                continue

            for digit in range(10):
                next_sum = square_sum + digit * digit
                next_counts[next_sum] = (next_counts[next_sum] + count) % MODULO
                next_values[next_sum] = (
                    next_values[next_sum] + value_sum + digit * place_value * count
                ) % MODULO

        counts = next_counts
        values = next_values
        place_value = (place_value * 10) % MODULO

    total = 0
    root = 1
    while root * root <= max_sum:
        total = (total + values[root * root]) % MODULO
        root += 1
    return total


def main() -> None:
    print(solve(20))


if __name__ == "__main__":
    main()
