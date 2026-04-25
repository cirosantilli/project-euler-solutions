#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path, PurePosixPath

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode


NUMERIC_RE = re.compile(r"^\d+(?:\.\d+)?$")


@dataclass(frozen=True)
class BenchmarkResult:
    path: str
    time_text: str
    runtime: Decimal


@dataclass(frozen=True)
class Improvement:
    problem: int
    reference: BenchmarkResult
    better: BenchmarkResult

    @property
    def seconds(self) -> Decimal:
        return self.reference.runtime - self.better.runtime


def scalar_text(node: Node) -> str | None:
    if not isinstance(node, ScalarNode):
        return None
    return node.value


def normalize_language(language: str) -> str:
    return language.removeprefix(".")


def has_glob_pattern(reference_set: str) -> bool:
    return any(char in reference_set for char in "*?[")


def is_reference_path(path: str, reference_set: str) -> bool:
    path_obj = PurePosixPath(path)
    reference_set = reference_set.rstrip("/")
    if has_glob_pattern(reference_set):
        return path_obj.match(reference_set)

    reference_path = PurePosixPath(reference_set)
    if reference_path.suffix:
        return path_obj == reference_path
    return path_obj.parent == reference_path


def parse_result(
    path_node: Node, time_node: Node, language: str
) -> BenchmarkResult | None:
    path = scalar_text(path_node)
    time_text = scalar_text(time_node)
    if path is None or time_text is None:
        return None
    if not path.endswith(f".{language}") or not NUMERIC_RE.fullmatch(time_text):
        return None
    return BenchmarkResult(path, time_text, Decimal(time_text))


def problem_improvement(
    problem_node: Node,
    results_node: Node,
    reference_set: str,
    language: str,
) -> Improvement | None:
    problem_text = scalar_text(problem_node)
    if problem_text is None:
        raise ValueError("benchmark.yaml contains a non-scalar problem id")
    problem = int(problem_text)

    if not isinstance(results_node, MappingNode):
        return None

    reference_results: list[BenchmarkResult] = []
    other_results: list[BenchmarkResult] = []
    for path_node, time_node in results_node.value:
        result = parse_result(path_node, time_node, language)
        if result is None:
            continue
        if is_reference_path(result.path, reference_set):
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


def load_benchmark(benchmark_path: Path) -> MappingNode | None:
    with benchmark_path.open() as benchmark_file:
        root = yaml.compose(benchmark_file, Loader=yaml.SafeLoader)

    if root is None:
        return None
    if not isinstance(root, MappingNode):
        raise ValueError(f"{benchmark_path} must contain a top-level mapping")
    return root


def format_seconds(seconds: Decimal) -> str:
    return f"{seconds:.3f}"


def print_greatest_improvements(
    benchmark_path: Path,
    reference_set: str,
    language: str,
) -> None:
    root = load_benchmark(benchmark_path)
    if root is None:
        return

    improvements = [
        improvement
        for problem_node, results_node in root.value
        if (
            improvement := problem_improvement(
                problem_node, results_node, reference_set, language
            )
        )
        is not None
    ]
    improvements.sort(
        key=lambda improvement: (-improvement.seconds, improvement.problem)
    )

    for improvement in improvements:
        print(
            "\t".join(
                [
                    str(improvement.problem),
                    format_seconds(improvement.seconds),
                    improvement.reference.path,
                    improvement.reference.time_text,
                    improvement.better.path,
                    improvement.better.time_text,
                ]
            )
        )


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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    language = normalize_language(args.language)
    reference_set = args.reference_set or f"solvers/*.{language}"
    try:
        print_greatest_improvements(args.benchmark, reference_set, language)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
