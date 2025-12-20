#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

import readme_tables

ROOT = Path(__file__).resolve().parent
README_PATH = ROOT / "README.adoc"
NUMERIC_RE = re.compile(r"^\d+(?:\.\d+)?$")
ID_RE = re.compile(r"(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot runtimes for working Python solvers from README.adoc."
    )
    parser.add_argument(
        "-o",
        "--output",
        default="plot.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot in a window after saving.",
    )
    return parser.parse_args()


def header_index(header_cells: list[str], label: str) -> int | None:
    target = label.strip().lower()
    for idx, cell in enumerate(header_cells):
        if cell.strip().lower() == target:
            return idx
    return None


def parse_id(cell: str) -> int | None:
    candidate = cell
    if "[" in cell and "]" in cell:
        candidate = cell[cell.find("[") + 1 : cell.rfind("]")]
    match = ID_RE.search(candidate)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def load_working_runtimes() -> list[tuple[int, float]]:
    lines = readme_tables.read_lines(README_PATH)
    rows = list(readme_tables.iter_table_rows(lines, "// RESULTS TABLE"))
    if not rows:
        raise RuntimeError("No rows found in RESULTS TABLE.")

    header_cells = rows[0][2]
    id_idx = header_index(header_cells, "ID")
    runtime_idx = header_index(header_cells, "Runtime (s)")
    error_idx = header_index(header_cells, "Error")
    if id_idx is None or runtime_idx is None or error_idx is None:
        raise RuntimeError("Missing required columns in RESULTS TABLE header.")

    results: list[tuple[int, float]] = []
    for _line_idx, _line, cells in rows[1:]:
        if len(cells) <= runtime_idx:
            continue
        error_cell = cells[error_idx].strip() if error_idx < len(cells) else ""
        if error_cell:
            continue
        runtime_cell = cells[runtime_idx].strip()
        if not NUMERIC_RE.match(runtime_cell):
            continue
        pid = parse_id(cells[id_idx].strip() if id_idx < len(cells) else "")
        if pid is None:
            continue
        results.append((pid, float(runtime_cell)))
    results.sort(key=lambda item: item[0])
    return results


def plot_runtimes(points: list[tuple[int, float]], output: str, show: bool) -> None:
    if not points:
        raise RuntimeError("No working solver runtimes found.")
    runtimes = np.array([rt for _pid, rt in points], dtype=float)

    bins = 50
    hist, edges = np.histogram(runtimes, bins=bins, density=False)
    widths = edges[1:] - edges[:-1]

    plt.figure(figsize=(10, 5))
    plt.bar(edges[:-1], hist, width=widths, align="edge", alpha=0.75, edgecolor="black")
    plt.yscale("log")
    plt.xlabel("Runtime (s)")
    plt.ylabel("Count (log scale)")
    plt.gca().xaxis.set_major_locator(MultipleLocator(50))
    plt.title("Runtime Distribution for Working Python Solvers (50 bins)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    if show:
        plt.show()


def main() -> None:
    args = parse_args()
    points = load_working_runtimes()
    plot_runtimes(points, args.output, args.show)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
