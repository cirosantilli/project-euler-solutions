#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from benchmark_utils import (
    BenchmarkEntry,
    is_path_in_set,
    iter_benchmark_entries,
    load_benchmark,
    normalize_language,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_LIMIT = 0
ALLOWED_LANGUAGES = {"py", "c", "cpp", "lean"}


@dataclass(frozen=True)
class SolverSet:
    label: str


@dataclass(frozen=True)
class SlowestEntry:
    solver_set: str
    benchmark: BenchmarkEntry


def solver_set_label(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        try:
            return path.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            return path.as_posix().rstrip("/")
    return path.as_posix().rstrip("/")


def parse_solver_sets(values: list[str] | None) -> list[SolverSet]:
    selected = values or ["solvers"]
    solver_sets: list[SolverSet] = []
    seen: set[str] = set()
    for value in selected:
        label = solver_set_label(value)
        if not label:
            raise ValueError(f"invalid solver set: {value}")
        if label in seen:
            continue
        seen.add(label)
        solver_sets.append(SolverSet(label))
    return solver_sets


def parse_lang_filter(values: list[str] | None) -> set[str]:
    if not values:
        return {"py"}
    selected: set[str] = set()
    for value in values:
        for token in value.split(","):
            token = normalize_language(token.strip())
            if not token:
                continue
            if token not in ALLOWED_LANGUAGES:
                raise ValueError(f"invalid language: {token}")
            selected.add(token)
    if not selected:
        raise ValueError("at least one language must be selected")
    return selected


def selected_slowest_entries(
    benchmark_path: Path,
    solver_sets: list[SolverSet],
    languages: set[str],
    limit: int,
) -> list[SlowestEntry]:
    root = load_benchmark(benchmark_path)
    if root is None:
        return []

    entries: list[SlowestEntry] = []
    for benchmark in iter_benchmark_entries(root, languages):
        for solver_set in solver_sets:
            if is_path_in_set(benchmark.path, solver_set.label):
                entries.append(SlowestEntry(solver_set.label, benchmark))
                break

    entries.sort(
        key=lambda entry: (
            -entry.benchmark.runtime,
            entry.benchmark.problem,
            entry.solver_set,
            entry.benchmark.path,
        )
    )
    if limit:
        entries = entries[:limit]
    return entries


def print_slowest_solvers(
    benchmark_path: Path,
    solver_sets: list[SolverSet],
    languages: set[str],
    limit: int,
) -> None:
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["problem", "solver_set", "language", "path", "runtime_seconds"])
    for entry in selected_slowest_entries(
        benchmark_path, solver_sets, languages, limit
    ):
        benchmark = entry.benchmark
        writer.writerow(
            [
                benchmark.problem,
                entry.solver_set,
                benchmark.language,
                benchmark.path,
                benchmark.time_text,
            ]
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print the slowest benchmarked solvers from benchmark.yaml as CSV."
        )
    )
    parser.add_argument(
        "benchmark",
        nargs="?",
        type=Path,
        default=Path("benchmark.yaml"),
        help="benchmark YAML file to read",
    )
    parser.add_argument(
        "--set",
        dest="sets",
        action="append",
        default=None,
        help=(
            "Select a solver set directory, e.g. --set solvers/eulersolve. "
            "Repeatable. Defaults to solvers."
        ),
    )
    parser.add_argument(
        "-l",
        "--lang",
        "--language",
        dest="langs",
        action="append",
        default=None,
        help=(
            "Limit to one or more languages (py,c,cpp,lean). Repeatable. "
            "Defaults to py."
        ),
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="maximum number of rows to print; defaults to 0 for all rows",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        solver_sets = parse_solver_sets(args.sets)
        languages = parse_lang_filter(args.langs)
        if args.limit < 0:
            raise ValueError("limit must be non-negative")
        print_slowest_solvers(args.benchmark, solver_sets, languages, args.limit)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
