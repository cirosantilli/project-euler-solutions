#!/usr/bin/env python
from __future__ import annotations


def name_value(name: str) -> int:
    """
    Alphabetical value: A=1, ..., Z=26.
    """
    return sum((ord(c) - ord("A") + 1) for c in name)


def load_names() -> list[str]:
    path = "0022_names.txt"
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    # File format: "MARY","PATRICIA",...
    parts = raw.split(",")
    return [p.strip().strip('"') for p in parts if p.strip()]


def names_are_uppercase(names: list[str]) -> bool:
    """
    Check the A–Z input domain, allowing empty names and an empty list.
    """
    for name in names:
        for c in name:
            if not 65 <= ord(c) <= 90:
                return False
    return True


def total_name_scores(names: list[str]) -> int:
    """
    @formalsw: Py from project_euler_statements.p22 import naive
    @formalsw: Requires(names_are_uppercase(names))
    @formalsw: Equiv(naive)
    """
    names_sorted = sorted(names)
    total = 0
    for i, nm in enumerate(names_sorted, start=1):
        total += i * name_value(nm)
    return total


def main() -> None:
    names = load_names()
    names_sorted = sorted(names)

    # Given example: COLIN is 938th and has score 49714 (in the provided dataset).
    colin_pos = names_sorted.index("COLIN") + 1
    colin_score = colin_pos * name_value("COLIN")
    assert colin_pos == 938
    assert colin_score == 49714

    ans = total_name_scores(names)

    print(ans)


if __name__ == "__main__":
    main()
