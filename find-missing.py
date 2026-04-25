#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_LANGUAGE = "py"
DEFAULT_SET = Path("solvers")
NON_SOLVER_EXTENSIONS = {"html", "json", "md", "out"}


def normalize_language(value: str) -> str:
    language = value.strip().removeprefix(".").lower()
    if not language:
        raise ValueError("language must not be empty")
    if "." in language or "/" in language or "\\" in language:
        raise ValueError(f"invalid language extension: {value}")
    return language


def solver_id(path: Path) -> int | None:
    if not path.is_file():
        return None
    if not path.stem.isdigit():
        return None

    extension = path.suffix.removeprefix(".").lower()
    if not extension or extension in NON_SOLVER_EXTENSIONS:
        return None
    return int(path.stem)


def solver_ids(solver_set: Path) -> set[int]:
    if not solver_set.exists():
        raise ValueError(f"solver set does not exist: {solver_set}")
    if not solver_set.is_dir():
        raise ValueError(f"solver set is not a directory: {solver_set}")

    ids: set[int] = set()
    for path in solver_set.iterdir():
        problem_id = solver_id(path)
        if problem_id is not None:
            ids.add(problem_id)
    return ids


def language_solver_ids(solver_set: Path, language: str) -> set[int]:
    return {
        int(path.stem)
        for path in solver_set.iterdir()
        if path.is_file()
        and path.stem.isdigit()
        and path.suffix.removeprefix(".").lower() == language
    }


def missing_solvers(solver_set: Path, language: str) -> list[Path]:
    existing_ids = solver_ids(solver_set)
    if not existing_ids:
        return []

    max_id = max(existing_ids)
    language_ids = language_solver_ids(solver_set, language)
    return [
        solver_set / f"{problem_id}.{language}"
        for problem_id in range(1, max_id + 1)
        if problem_id not in language_ids
    ]


def selected_language(args: argparse.Namespace) -> str:
    values = [value for value in (args.language, args.language_option) if value]
    if not values:
        return DEFAULT_LANGUAGE

    languages = {normalize_language(value) for value in values}
    if len(languages) != 1:
        raise ValueError("language was provided more than once with different values")
    return languages.pop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print missing solver source paths for a language."
    )
    parser.add_argument(
        "language",
        nargs="?",
        help="language extension to check, e.g. py, cpp, or .lean; defaults to py",
    )
    parser.add_argument(
        "-l",
        "--lang",
        "--language",
        dest="language_option",
        help="language extension to check; overrides the positional default",
    )
    parser.add_argument(
        "--set",
        dest="sets",
        action="append",
        type=Path,
        help=(
            "solver set directory to scan, e.g. --set solvers/eulersolve; "
            "repeatable; defaults to solvers"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        language = selected_language(args)
        solver_sets = args.sets or [DEFAULT_SET]
        for solver_set in solver_sets:
            for path in missing_solvers(solver_set, language):
                print(path)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
