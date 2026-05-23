#!/usr/bin/env python


def reverse_number(value: int) -> int:
    result = 0
    while value:
        result = 10 * result + value % 10
        value //= 10
    return result


def only_odd(value: int) -> bool:
    while value:
        if value % 2 == 0:
            return False
        value //= 10
    return True


def brute_count_range(start: int, stop: int) -> int:
    count = 0
    for value in range(start, stop):
        if value % 10 != 0 and only_odd(value + reverse_number(value)):
            count += 1
    return count


def count_by_length(length: int) -> int:
    if length % 2 == 0:
        return 20 * 30 ** (length // 2 - 1)
    if length % 4 == 3:
        return 100 * 500 ** ((length - 3) // 4)
    return 0


def count_reversible(limit: int) -> int:
    total = 0
    next_power = 10
    length = 1

    while next_power <= limit:
        total += count_by_length(length)
        next_power *= 10
        length += 1

    previous_power = next_power // 10
    if previous_power < limit:
        total += brute_count_range(previous_power, limit)

    return total


def main() -> None:
    assert count_reversible(1000) == 120
    print(count_reversible(1_000_000_000))


if __name__ == "__main__":
    main()
