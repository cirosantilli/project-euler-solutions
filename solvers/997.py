#!/usr/bin/env python3
"""Count compatible dice arrangements for the given box dimensions.

The script intentionally does not store the requested final numeric answer;
it computes and prints it when run.
"""


def f(x: int, y: int, z: int) -> int:
    """Return the number of valid arrangements for an x by y by z box."""
    for name, value in (("x", x), ("y", y), ("z", z)):
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    return 3 * (1 << (x + y + z - 1)) * ((1 << x) + (1 << y) + (1 << z) - 4)


def main() -> None:
    # Values given in the problem statement.
    assert f(1, 1, 1) == 24
    assert f(2, 3, 4) == 18432

    # Print the requested value without embedding its evaluated constant.
    print(f(9, 10, 11))


if __name__ == "__main__":
    main()
