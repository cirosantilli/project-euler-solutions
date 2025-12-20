from __future__ import annotations

from pathlib import Path
from typing import Iterable


def read_lines(path: Path) -> list[str]:
    return path.read_text().splitlines()


def find_table_block(lines: list[str], marker: str) -> tuple[int, int]:
    try:
        marker_idx = next(i for i, line in enumerate(lines) if line.strip() == marker)
    except StopIteration as exc:
        raise RuntimeError(f"Could not find {marker} marker in README.adoc") from exc

    start = None
    end = None
    for i in range(marker_idx + 1, len(lines)):
        if lines[i].strip() == "|===":
            if start is None:
                start = i
            else:
                end = i
                break
    if start is None or end is None:
        raise RuntimeError(f"Could not find table for {marker} in README.adoc")
    return start, end


def split_table_row(line: str) -> list[str]:
    if "|" not in line:
        return []
    parts = line.split("|")
    return [cell.strip() for cell in parts[1:]]


def iter_table_rows(
    lines: list[str], marker: str
) -> Iterable[tuple[int, str, list[str]]]:
    start, end = find_table_block(lines, marker)
    for i in range(start + 1, end):
        line = lines[i]
        cells = split_table_row(line)
        yield i, line, cells
