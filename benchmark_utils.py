from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode


NUMERIC_RE = re.compile(r"^\d+(?:\.\d+)?$")


@dataclass(frozen=True)
class BenchmarkEntry:
    problem: int
    path: str
    language: str
    time_text: str
    runtime: Decimal


def scalar_text(node: Node) -> str | None:
    if not isinstance(node, ScalarNode):
        return None
    return node.value


def normalize_language(language: str) -> str:
    return language.removeprefix(".").lower()


def benchmark_path_language(path: str) -> str | None:
    suffix = PurePosixPath(path).suffix
    if not suffix:
        return None
    return normalize_language(suffix[1:])


def has_glob_pattern(path_pattern: str) -> bool:
    return any(char in path_pattern for char in "*?[")


def is_path_in_set(path: str, solver_set: str) -> bool:
    path_obj = PurePosixPath(path)
    solver_set = solver_set.rstrip("/")
    if has_glob_pattern(solver_set):
        return path_obj.match(solver_set)

    solver_set_path = PurePosixPath(solver_set)
    if solver_set_path.suffix:
        return path_obj == solver_set_path
    return path_obj.parent == solver_set_path


def load_benchmark(benchmark_path: Path) -> MappingNode | None:
    with benchmark_path.open() as benchmark_file:
        root = yaml.compose(benchmark_file, Loader=yaml.SafeLoader)

    if root is None:
        return None
    if not isinstance(root, MappingNode):
        raise ValueError(f"{benchmark_path} must contain a top-level mapping")
    return root


def iter_benchmark_entries(
    root: MappingNode, languages: Iterable[str] | None = None
) -> Iterator[BenchmarkEntry]:
    selected_languages = (
        {normalize_language(language) for language in languages}
        if languages is not None
        else None
    )
    for problem_node, results_node in root.value:
        problem_text = scalar_text(problem_node)
        if problem_text is None:
            raise ValueError("benchmark.yaml contains a non-scalar problem id")
        problem = int(problem_text)

        if not isinstance(results_node, MappingNode):
            continue

        for path_node, time_node in results_node.value:
            path = scalar_text(path_node)
            time_text = scalar_text(time_node)
            if path is None or time_text is None:
                continue
            language = benchmark_path_language(path)
            if language is None:
                continue
            if selected_languages is not None and language not in selected_languages:
                continue
            if not NUMERIC_RE.fullmatch(time_text):
                continue
            yield BenchmarkEntry(
                problem=problem,
                path=path,
                language=language,
                time_text=time_text,
                runtime=Decimal(time_text),
            )


def format_seconds(seconds: Decimal) -> str:
    return f"{seconds:.3f}"
