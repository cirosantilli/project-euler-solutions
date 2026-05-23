#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import yaml

from benchmark_utils import (
    BenchmarkEntry,
    format_seconds,
    is_path_in_set,
    iter_benchmark_entries,
    load_benchmark,
    normalize_language,
)


@dataclass(frozen=True)
class Improvement:
    problem: int
    reference: BenchmarkEntry
    better: BenchmarkEntry

    @property
    def seconds(self) -> Decimal:
        return self.reference.runtime - self.better.runtime

    @property
    def ratio(self) -> Decimal:
        return self.better.runtime / self.reference.runtime


def problem_improvement(
    problem: int,
    entries: list[BenchmarkEntry],
    reference_set: str,
) -> Improvement | None:
    reference_results: list[BenchmarkEntry] = []
    other_results: list[BenchmarkEntry] = []
    for result in entries:
        if is_path_in_set(result.path, reference_set):
            reference_results.append(result)
        else:
            other_results.append(result)

    if not reference_results or not other_results:
        return None

    reference = min(reference_results, key=lambda result: (result.runtime, result.path))
    better = min(other_results, key=lambda result: (result.runtime, result.path))
    if better.runtime >= reference.runtime:
        return None
    return Improvement(problem, reference, better)


def print_greatest_improvements(
    benchmark_path: Path,
    reference_set: str,
    language: str,
    max_ratio: Decimal | None,
) -> None:
    root = load_benchmark(benchmark_path)
    if root is None:
        return

    entries_by_problem: dict[int, list[BenchmarkEntry]] = {}
    for entry in iter_benchmark_entries(root, [language]):
        entries_by_problem.setdefault(entry.problem, []).append(entry)

    improvements = [
        improvement
        for problem, entries in entries_by_problem.items()
        if (improvement := problem_improvement(problem, entries, reference_set))
        is not None
    ]
    if max_ratio is not None:
        improvements = [
            improvement
            for improvement in improvements
            if improvement.ratio <= max_ratio
        ]
    improvements.sort(
        key=lambda improvement: (-improvement.seconds, improvement.problem)
    )

    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(
        [
            "problem",
            "improvement_seconds",
            "ratio",
            "reference_path",
            "reference_runtime_seconds",
            "better_path",
            "better_runtime_seconds",
        ]
    )
    for improvement in improvements:
        writer.writerow(
            [
                improvement.problem,
                format_seconds(improvement.seconds),
                f"{improvement.ratio:.2f}",
                improvement.reference.path,
                improvement.reference.time_text,
                improvement.better.path,
                improvement.better.time_text,
            ]
        )


def parse_decimal(value: str) -> Decimal:
    try:
        decimal = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"invalid decimal value: {value}") from exc
    if not decimal.is_finite():
        raise argparse.ArgumentTypeError(f"invalid decimal value: {value}")
    return decimal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print problems where a benchmarked solver beats the selected "
            "reference set, sorted by greatest improvement."
        ),
    )
    parser.add_argument(
        "benchmark",
        nargs="?",
        type=Path,
        default=Path("benchmark.yaml"),
        help="benchmark YAML file to read",
    )
    parser.add_argument(
        "--reference-set",
        default=None,
        help=(
            "reference directory, file, or glob. Defaults to solvers/*.LANG, "
            "for example solvers/*.py"
        ),
    )
    parser.add_argument(
        "--language",
        "-l",
        default="py",
        help="solver file extension to compare",
    )
    parser.add_argument(
        "--max-ratio",
        type=parse_decimal,
        default=None,
        help="only include improvements with better/reference runtime ratio at most this value",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    language = normalize_language(args.language)
    reference_set = args.reference_set or f"solvers/*.{language}"
    try:
        print_greatest_improvements(
            args.benchmark,
            reference_set,
            language,
            args.max_ratio,
        )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
