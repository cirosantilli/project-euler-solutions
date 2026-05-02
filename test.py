#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import lint
import readme_tables
import summary

ROOT = Path(__file__).resolve().parent
SOLUTIONS_PATH = ROOT / "data/projecteuler-solutions/Solutions.md"
SOLVERS_DIR = ROOT / "solvers"
BENCHMARK_PATH = ROOT / "benchmark.yaml"
STATEMENTS_DOCS_DIR = ROOT / "data" / "project-euler-statements" / "data" / "documents"
STATEMENTS_PROBLEM_DIR = ROOT / "data" / "project-euler-statements" / "data" / "problem"

LINE_RE = re.compile(r"^(\d+)\.\s+(.*)$")
NUMERIC_RE = re.compile(r"^\d+(?:\.\d+)?$")
EXPECTED_GOT_RE = re.compile(r"expected (.+), got (.+)")
LANG_HINT_RE = re.compile(r"^(\d+)\.(py|c|cpp|lean)$")
MISSING_REFERENCE_MESSAGE = "missing reference answer"
MISSING_REFERENCE_ERROR = f"error: {MISSING_REFERENCE_MESSAGE}"
TestId = int | str


@dataclass
class Result:
    puzzle_id: TestId
    correct: bool
    elapsed: float | None
    output: str | None
    message: str
    language: str | None
    source_path: Path | None
    reference_answer_checked: bool = True


@dataclass
class SolverTarget:
    puzzle_id: TestId
    path: Path
    language: str
    checks_reference_answer: bool = True
    suite: str = "solvers"


@dataclass(frozen=True)
class SolverSet:
    label: str
    path: Path


EXPLICIT_ONLY_TESTS: dict[str, SolverTarget] = {
    "secret": SolverTarget(
        puzzle_id="secret",
        path=SOLVERS_DIR / "secret.py",
        language="py",
        checks_reference_answer=False,
    ),
}
EXPLICIT_ONLY_PATHS = {
    target.path.resolve(): name for name, target in EXPLICIT_ONLY_TESTS.items()
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Project Euler solvers and verify their answers."
    )
    parser.add_argument(
        "ids",
        nargs="*",
        type=str,
        help=(
            "IDs, ranges, or explicit-only test names to run "
            "(e.g. 2-4 6 secret). Defaults to every standard solver."
        ),
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=float,
        default=600,
        help="Per-solver timeout in seconds.",
    )
    parser.add_argument(
        "-A",
        "--autoupdate",
        action="store_true",
        help="Update README.adoc results table with the latest run.",
    )
    parser.add_argument(
        "--autoupdate-not-found",
        action="store_true",
        help="Update README.adoc to mark only missing solvers (no runs).",
    )
    parser.add_argument(
        "--autoupdate-links",
        action="store_true",
        help="Update README.adoc to add missing .md links (no runs).",
    )
    parser.add_argument(
        "-l",
        "--lang",
        action="append",
        default=None,
        help="Limit to one or more languages (py,c,cpp,lean). Repeatable.",
    )
    parser.add_argument(
        "--set",
        dest="sets",
        action="append",
        default=None,
        help=(
            "Run a solver set from a directory, e.g. --set solvers/eulersolve. "
            "Repeatable. Defaults to Python unless --lang is given."
        ),
    )
    parser.add_argument(
        "-u",
        "--uncommitted",
        action="store_true",
        help="Run only solvers modified or added since the last git commit.",
    )
    return parser.parse_args()


def test_id_sort_key(value: TestId) -> tuple[int, int, str]:
    if isinstance(value, int):
        return (0, value, "")
    return (1, 0, value)


def sort_test_ids(values: list[TestId]) -> list[TestId]:
    return sorted(set(values), key=test_id_sort_key)


def table_sort_key(
    test_id: TestId, language: str = "", checks_reference_answer: bool = True
) -> tuple[int, int, int, str, str]:
    if not checks_reference_answer:
        return (1, 0, 0, str(test_id), language)
    type_rank, numeric_id, text_id = test_id_sort_key(test_id)
    return (0, type_rank, numeric_id, text_id, language)


def checks_reference_answer_for_id(test_id: TestId) -> bool:
    if isinstance(test_id, str):
        target = EXPLICIT_ONLY_TESTS.get(test_id)
        if target is not None:
            return target.checks_reference_answer
    return True


def row_sort_key(key: tuple[TestId, str]) -> tuple[int, int, int, str, str]:
    test_id, language = key
    return table_sort_key(test_id, language, checks_reference_answer_for_id(test_id))


def result_sort_key(res: Result) -> tuple[int, int, int, str, str]:
    return table_sort_key(
        res.puzzle_id, res.language or "", res.reference_answer_checked
    )


def resolve_cli_path(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    if "/" not in value and path.parent == Path("."):
        candidate = SOLVERS_DIR / value
        if candidate.exists():
            return candidate
    return path


def explicit_only_name_for_path(path: Path) -> str | None:
    return EXPLICIT_ONLY_PATHS.get(path.resolve())


def test_id_from_link_target(link_target: str) -> TestId | None:
    path = Path(link_target)
    pid = parse_solver_id(path)
    if pid is not None:
        return pid
    return explicit_only_name_for_path(path)


def expand_ids(values: list[str]) -> tuple[list[TestId], dict[TestId, list[Path]]]:
    ids: list[TestId] = []
    overrides: dict[TestId, list[Path]] = {}
    for value in values:
        if value in EXPLICIT_ONLY_TESTS:
            ids.append(value)
            continue
        if (
            "_" in value
            and not value.endswith(".py")
            and not value.endswith(".out")
            and "/" not in value
        ):
            candidate = SOLVERS_DIR / f"{value}.py"
            if not candidate.exists():
                raise ValueError(f"solver not found: {candidate}")
            pid = parse_solver_id(candidate)
            if pid is None:
                raise ValueError(f"invalid solver path: {candidate}")
            ids.append(pid)
            overrides.setdefault(pid, []).append(candidate)
            continue
        if value.endswith(".py") or value.endswith(".out") or "/" in value:
            path = resolve_cli_path(value)
            if not path.exists():
                raise ValueError(f"solver not found: {value}")
            explicit_name = explicit_only_name_for_path(path)
            if explicit_name is not None:
                ids.append(explicit_name)
                continue
            pid = parse_solver_id(path)
            if pid is None:
                raise ValueError(f"invalid solver path: {value}")
            ids.append(pid)
            overrides.setdefault(pid, []).append(path)
            continue
        if "-" in value:
            start_str, sep, end_str = value.partition("-")
            if not sep:
                raise ValueError(f"invalid range: {value}")
            try:
                start = int(start_str)
                end = int(end_str)
            except ValueError as exc:
                raise ValueError(f"invalid range: {value}") from exc
            if start > end:
                raise ValueError(f"invalid range: {value}")
            ids.extend(range(start, end + 1))
        else:
            try:
                ids.append(int(value))
            except ValueError as exc:
                raise ValueError(f"invalid id: {value}") from exc
    return ids, overrides


def extract_lang_hints(values: list[str]) -> tuple[list[str], set[str]]:
    remaining: list[str] = []
    hints: set[str] = set()
    for value in values:
        if "/" in value or Path(value).exists():
            remaining.append(value)
            continue
        match = LANG_HINT_RE.match(value)
        if match:
            remaining.append(match.group(1))
            hints.add(match.group(2))
            continue
        remaining.append(value)
    return remaining, hints


def load_reference_answers() -> dict[int, str]:
    solutions: dict[int, str] = {}
    with SOLUTIONS_PATH.open() as fh:
        for line in fh:
            line = line.strip()
            match = LINE_RE.match(line)
            if not match:
                continue
            idx = int(match.group(1))
            solutions[idx] = match.group(2).strip()
    return solutions


def load_statement_ids() -> set[int]:
    ids: set[int] = set()
    for path in STATEMENTS_PROBLEM_DIR.glob("*.html"):
        try:
            ids.add(int(path.stem))
        except ValueError:
            continue
    return ids


def collect_explicit_only_targets(
    lang_filter: set[str] | None,
) -> dict[str, SolverTarget]:
    targets: dict[str, SolverTarget] = {}
    for name, target in EXPLICIT_ONLY_TESTS.items():
        if lang_filter and target.language not in lang_filter:
            continue
        targets[name] = target
    return targets


def targets_for_id(
    test_id: TestId,
    solver_targets: dict[int, list[SolverTarget]],
    explicit_only_targets: dict[str, SolverTarget],
) -> list[SolverTarget]:
    if isinstance(test_id, int):
        return solver_targets.get(test_id, [])
    target = explicit_only_targets.get(test_id)
    return [target] if target is not None else []


def collect_solver_targets(
    lang_filter: set[str] | None,
) -> dict[int, list[SolverTarget]]:
    targets: dict[int, list[SolverTarget]] = {}
    for path in SOLVERS_DIR.glob("*.py"):
        if explicit_only_name_for_path(path) is not None:
            continue
        lang = detect_language(path)
        if lang is None or (lang_filter and lang not in lang_filter):
            continue
        pid = parse_solver_id(path)
        if pid is None:
            continue
        targets.setdefault(pid, []).append(SolverTarget(pid, path, lang))
    for path in SOLVERS_DIR.glob("*.out"):
        if explicit_only_name_for_path(path) is not None:
            continue
        lang = detect_language(path)
        if lang is None or (lang_filter and lang not in lang_filter):
            continue
        pid = parse_solver_id(path)
        if pid is None:
            continue
        targets.setdefault(pid, []).append(SolverTarget(pid, path, lang))
    order = {"py": 0, "c": 1, "cpp": 2, "lean": 3}
    for entries in targets.values():
        entries.sort(key=lambda item: order.get(item.language, 99))
    return targets


def solver_set_label(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def parse_solver_sets(values: list[str] | None) -> list[SolverSet]:
    if not values:
        return []
    solver_sets: list[SolverSet] = []
    seen: set[Path] = set()
    for value in values:
        path = Path(value)
        if not path.is_absolute():
            path = ROOT / path
        path = path.resolve()
        if not path.is_dir():
            raise ValueError(f"solver set not found: {value}")
        if path in seen:
            continue
        seen.add(path)
        solver_sets.append(SolverSet(solver_set_label(path), path))
    return solver_sets


def collect_solver_set_targets(
    solver_set: SolverSet,
    lang_filter: set[str] | None,
) -> dict[int, list[SolverTarget]]:
    selected = lang_filter or {"py"}
    targets: dict[int, list[SolverTarget]] = {}
    if "py" in selected:
        for path in solver_set.path.glob("*.py"):
            if path.resolve().parent == SOLVERS_DIR.resolve():
                if explicit_only_name_for_path(path) is not None:
                    continue
            pid = parse_solver_id(path)
            if pid is None:
                continue
            targets.setdefault(pid, []).append(
                SolverTarget(pid, path, "py", suite=solver_set.label)
            )
    for path in solver_set.path.glob("*.out"):
        lang = detect_language(path)
        if lang is None or lang not in selected:
            continue
        pid = parse_solver_id(path)
        if pid is None:
            continue
        targets.setdefault(pid, []).append(
            SolverTarget(pid, path, lang, suite=solver_set.label)
        )
    order = {"py": 0, "c": 1, "cpp": 2, "lean": 3}
    for entries in targets.values():
        entries.sort(key=lambda item: (order.get(item.language, 99), item.path.name))
    return targets


def expand_solver_set_ids(values: list[str]) -> list[int]:
    ids: list[int] = []
    for value in values:
        if value.endswith(".py") or value.endswith(".out") or "/" in value:
            raise ValueError("--set mode accepts numeric IDs and ranges, not paths")
        if "-" in value:
            start_str, sep, end_str = value.partition("-")
            if not sep:
                raise ValueError(f"invalid range: {value}")
            try:
                start = int(start_str)
                end = int(end_str)
            except ValueError as exc:
                raise ValueError(f"invalid range: {value}") from exc
            if start > end:
                raise ValueError(f"invalid range: {value}")
            ids.extend(range(start, end + 1))
            continue
        try:
            ids.append(int(value))
        except ValueError as exc:
            raise ValueError(f"invalid id: {value}") from exc
    return sorted(set(ids))


def collect_solver_entries() -> dict[tuple[int, str], Path]:
    entries: dict[tuple[int, str], Path] = {}
    for path in SOLVERS_DIR.glob("*"):
        if explicit_only_name_for_path(path) is not None:
            continue
        language = detect_language(path)
        if language is None:
            continue
        pid = parse_solver_id(path)
        if pid is None:
            continue
        key = (pid, language)
        source_path = path
        if path.suffix == ".out":
            candidate = source_from_target(path, language)
            if candidate.exists():
                source_path = candidate
        existing = entries.get(key)
        if existing is None:
            entries[key] = source_path
            continue
        if existing.suffix == ".out" and source_path.suffix != ".out":
            entries[key] = source_path
    return entries


def collect_uncommitted_solvers() -> list[int]:
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain", "--", str(SOLVERS_DIR)],
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError("Failed to run git to find uncommitted solvers.") from exc
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "git status failed.")
    ids: list[int] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if not parts:
            continue
        path_str = parts[-1]
        path = Path(path_str)
        if path.parent.name != "solvers":
            continue
        pid = parse_solver_id(path)
        if pid is not None:
            ids.append(pid)
    return sorted(set(ids))


def parse_solver_id(path: Path) -> int | None:
    stem = path.stem
    if stem.isdigit():
        return int(stem)
    if "_" in stem:
        prefix = stem.split("_", 1)[0]
        if prefix.isdigit():
            return int(prefix)
    return None


def detect_language(path: Path) -> str | None:
    if path.suffix == ".py":
        return "py"
    if path.suffix == ".c":
        return "c"
    if path.suffix == ".cpp":
        return "cpp"
    if path.suffix == ".lean":
        return "lean"
    if path.suffix == ".out":
        if path.stem.endswith("_c"):
            return "c"
        if path.stem.endswith("_cpp"):
            return "cpp"
        if path.stem.endswith("_lean"):
            return "lean"
    return None


def source_from_target(path: Path, language: str) -> Path:
    if language == "py":
        return path
    if language == "c":
        stem = path.stem.removesuffix("_c")
        return path.with_name(f"{stem}.c")
    if language == "cpp":
        stem = path.stem.removesuffix("_cpp")
        return path.with_name(f"{stem}.cpp")
    if language == "lean":
        stem = path.stem.removesuffix("_lean")
        return path.with_name(f"{stem}.lean")
    return path


def is_primary_python_solver_path(path: Path) -> bool:
    if path.suffix != ".py":
        return False
    full_path = path if path.is_absolute() else ROOT / path
    return full_path.resolve().parent == SOLVERS_DIR.resolve()


def is_primary_python_result(res: Result) -> bool:
    language = res.language or "py"
    if language != "py":
        return False
    if res.source_path is None:
        return True
    return is_primary_python_solver_path(res.source_path)


def target_from_source(path: Path, language: str) -> Path:
    if language in {"c", "cpp", "lean"} and path.suffix != ".out":
        return path.with_name(f"{path.stem}_{language}.out")
    return path


def run_solver(
    path: Path, timeout: float | None, language: str
) -> tuple[int | None, str, str, float, bool]:
    solver_path = target_from_source(path, language).resolve()
    start = time.perf_counter()
    try:
        if language == "py":
            command = ["pypy3", str(solver_path)]
        else:
            command = [str(solver_path)]
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=STATEMENTS_DOCS_DIR,
            timeout=timeout,
        )
        elapsed = time.perf_counter() - start
        return proc.returncode, proc.stdout, proc.stderr, elapsed, False
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        return None, exc.stdout or "", exc.stderr or "", elapsed, True


def format_stdout_for_display(stdout: str) -> str:
    if not stdout:
        return ""
    if stdout.endswith("\n"):
        stdout = stdout[:-1]
    return stdout.replace("\n", "\\n")


def stdout_matches_expected(stdout: str, expected: str) -> bool:
    return stdout == expected or stdout == f"{expected}\n"


def wrong_answer_detail(display_output: str, expected: str | None) -> str:
    return f"got {display_output!r} expected {expected!r}"


def format_row(res: Result) -> str:
    time_cell = f"{res.elapsed:.3f}" if res.elapsed is not None else ""
    output_cell = format_output_cell(res)
    error_cell = "" if res.correct else normalize_error_cell(res.message)
    link = format_id_cell(res)
    explanation_cell = explanation_link(res.puzzle_id)
    return format_row_fields(
        link,
        explanation_cell,
        time_cell,
        output_cell,
        error_cell,
    )


def format_row_other(res: Result) -> str:
    time_cell = f"{res.elapsed:.3f}" if res.elapsed is not None else ""
    output_cell = format_output_cell(res)
    error_cell = "" if res.correct else normalize_error_cell(res.message)
    link = format_id_cell(res)
    return format_row_fields_other(link, time_cell, output_cell, error_cell)


def format_id_cell(res: Result) -> str:
    if res.message == "solver not found":
        if res.language and res.language != "py":
            return f"{res.puzzle_id}.{res.language}"
        return f"{res.puzzle_id}.py"
    link_path = (
        res.source_path
        if res.source_path is not None
        else SOLVERS_DIR / f"{res.puzzle_id}.py"
    )
    try:
        rel_path = link_path.resolve().relative_to(ROOT)
    except ValueError:
        rel_path = link_path
    return f"link:{rel_path.as_posix()}[{rel_path.name}]"


def result_key(res: Result) -> tuple[TestId, str]:
    if res.source_path is not None:
        link_path = res.source_path
    else:
        link_path = SOLVERS_DIR / f"{res.puzzle_id}.py"
    language = res.language or detect_language(link_path) or ""
    return res.puzzle_id, language


def explanation_link(puzzle_id: TestId) -> str:
    md_path = SOLVERS_DIR / f"{puzzle_id}.md"
    if not md_path.exists():
        return ""
    try:
        rel_path = md_path.resolve().relative_to(ROOT)
    except ValueError:
        rel_path = md_path
    return f"link:{rel_path.as_posix()}[{md_path.name}]"


def looks_numeric(value: str) -> bool:
    return bool(NUMERIC_RE.match(value))


def looks_like_model_cell(value: str) -> bool:
    value = value.strip().lower()
    return value.startswith(
        ("gpt-", "o1", "o3", "o4", "codex-", "claude-", "gemini-")
    )


def trim_trailing_empty_cells(cells: list[str]) -> list[str]:
    while len(cells) > 7 and cells[-1] == "":
        cells = cells[:-1]
    return cells


def row_has_statement(cells: list[str]) -> bool:
    if len(cells) < 2:
        return False
    if cells[1].startswith("link:"):
        return True
    if looks_numeric(cells[1]):
        return False
    if len(cells) >= 3 and (looks_numeric(cells[2]) or cells[2] == ""):
        return True
    return False


def normalize_row_fields(
    pid: TestId, cells: list[str]
) -> tuple[str, str, str, str, str] | None:
    if (
        len(cells) >= 7
        and cells[1].startswith("link:")
        and cells[2].startswith("link:")
    ):
        cells = [cells[0], cells[1], *cells[3:]]
    if len(cells) >= 7:
        (
            id_cell,
            statement_cell,
            time_cell,
            _model_cell,
            _tokens_cell,
            output_cell,
            error_cell,
        ) = cells[:7]
    elif len(cells) == 6:
        (
            id_cell,
            statement_cell,
            time_cell,
            _model_cell,
            _tokens_cell,
            error_cell,
        ) = cells
        output_cell = ""
    elif len(cells) == 5:
        if row_has_statement(cells):
            id_cell, statement_cell, time_cell, fourth_cell, fifth_cell = cells
            if looks_like_model_cell(fourth_cell) or (
                not fourth_cell and looks_numeric(fifth_cell)
            ):
                output_cell = ""
                error_cell = (
                    fifth_cell if fifth_cell and not looks_numeric(fifth_cell) else ""
                )
            else:
                output_cell = fourth_cell
                error_cell = fifth_cell
        else:
            id_cell, time_cell, _model_cell, _tokens_cell, error_cell = cells
            statement_cell = explanation_link(pid)
            output_cell = ""
    elif len(cells) == 4:
        id_cell, time_cell, _model_cell, _tokens_cell = cells
        statement_cell = explanation_link(pid)
        output_cell = ""
        error_cell = ""
    else:
        return None
    if not statement_cell:
        statement_cell = explanation_link(pid)
    output_cell, error_cell = normalize_output_error_cells(output_cell, error_cell)
    return (
        id_cell,
        statement_cell,
        time_cell,
        output_cell,
        error_cell,
    )


def normalize_error_cell(value: str) -> str:
    if not value:
        return ""
    if value.startswith("error: "):
        return value
    return f"error: {value}"


def strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def normalize_output_error_cells(output_cell: str, error_cell: str) -> tuple[str, str]:
    output_cell = output_cell.strip()
    error_cell = error_cell.strip()
    if not error_cell:
        return output_cell, ""
    normalized_error = error_cell
    if normalized_error.startswith("error: "):
        normalized_error = normalized_error[len("error: ") :]
    match = EXPECTED_GOT_RE.match(normalized_error)
    if match:
        expected, got = match.groups()
        expected = strip_wrapping_quotes(expected.strip())
        got = strip_wrapping_quotes(got.strip())
        if not output_cell:
            output_cell = got
        elif output_cell.startswith("got "):
            output_cell = strip_wrapping_quotes(output_cell[len("got ") :].strip())
        else:
            output_cell = strip_wrapping_quotes(output_cell)
        error_cell = normalize_error_cell(f"expected {expected}")
    else:
        error_cell = normalize_error_cell(error_cell)
    return output_cell, error_cell


def format_output_cell(res: Result) -> str:
    if not res.reference_answer_checked and res.correct:
        return ""
    if res.output is None or res.output == "":
        return ""
    if not res.correct and res.message.startswith("expected "):
        return res.output
    return res.output


def format_row_fields(
    id_cell: str,
    statement_cell: str,
    time_cell: str,
    output_cell: str,
    error_cell: str,
) -> str:
    return (
        f"| {id_cell} | {statement_cell} | {time_cell} | "
        f"{output_cell} | {error_cell}"
    ).rstrip()


def format_row_fields_other(
    id_cell: str,
    time_cell: str,
    output_cell: str,
    error_cell: str,
) -> str:
    return f"| {id_cell} | {time_cell} | {output_cell} | {error_cell}".rstrip()


def is_missing_reference_error_cell(error_cell: str) -> bool:
    normalized = normalize_error_cell(error_cell.strip())
    return normalized == MISSING_REFERENCE_ERROR


def has_comparison_runtime(res: Result) -> bool:
    return res.correct or res.message == MISSING_REFERENCE_MESSAGE


def comparison_runtime_cell(res: Result) -> str | None:
    if res.elapsed is None or not has_comparison_runtime(res):
        return None
    return f"{res.elapsed:.3f}"


def find_marker_index(lines: list[str], marker: str) -> int:
    try:
        return next(i for i, line in enumerate(lines) if line.strip() == marker)
    except StopIteration as exc:
        raise RuntimeError(f"Could not find {marker} marker in README.adoc") from exc


def marker_section_limit(lines: list[str], marker_idx: int) -> int:
    for i in range(marker_idx + 1, len(lines)):
        stripped = lines[i].strip()
        if stripped.startswith("// ") or stripped.startswith("=="):
            return i
    return len(lines)


def link_target_from_cell(cell: str) -> str | None:
    link_match = re.search(r"(?:link:)?([^\[]+)\[[^\]]*\]", cell)
    if not link_match:
        return None
    return link_match.group(1).strip()


def parse_id_cell(cell: str) -> int | None:
    link_target = link_target_from_cell(cell)
    if link_target is not None:
        pid = test_id_from_link_target(link_target)
        return pid if isinstance(pid, int) else None
    plain = cell.strip()
    if plain.isdigit():
        return int(plain)
    plain_match = re.match(r"^(\d+)\.(?:py|c|cpp|lean)$", plain)
    if plain_match:
        return int(plain_match.group(1))
    return None


def parse_primary_python_row_map(lines: list[str]) -> dict[tuple[TestId, str], str]:
    start, end = readme_tables.find_table_block(lines, "// RESULTS TABLE")
    row_re = re.compile(r"^\|\s+link:([^\[]+)\[")
    plain_re = re.compile(r"^\|\s+(\d+)\.py\s+\|")
    row_map: dict[tuple[TestId, str], str] = {}
    for i in range(start + 1, end):
        line = lines[i]
        match = row_re.match(line)
        if match:
            link_target = match.group(1)
            path = Path(link_target)
            if not is_primary_python_solver_path(path):
                continue
            pid = test_id_from_link_target(link_target)
            if pid is None:
                continue
            language = detect_language(path) or ""
        else:
            plain_match = plain_re.match(line)
            if not plain_match:
                continue
            pid = int(plain_match.group(1))
            language = "py"
        cells = trim_trailing_empty_cells(readme_tables.split_table_row(line))
        normalized = normalize_row_fields(pid, cells)
        if normalized is None:
            continue
        (
            id_cell,
            statement_cell,
            time_cell,
            output_cell,
            error_cell,
        ) = normalized
        row_map[(pid, language)] = format_row_fields(
            id_cell,
            statement_cell,
            time_cell,
            output_cell,
            error_cell,
        )
    return row_map


def other_results_block_span(lines: list[str]) -> tuple[int, int]:
    marker_idx = find_marker_index(lines, "// BENCHMARK TABLE")
    limit = marker_section_limit(lines, marker_idx)
    start = marker_idx + 1
    while start < limit and not lines[start].strip():
        start += 1
    if start < limit and lines[start].strip() == "|===":
        table_start, table_end = readme_tables.find_table_block(
            lines, "// BENCHMARK TABLE"
        )
        return table_start, table_end + 1
    end = start
    while end < limit and lines[end].strip().startswith("*"):
        end += 1
    return start, end


def relative_readme_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def other_result_path_for_suite(pid: int, suite_label: str, language: str) -> str:
    return f"{suite_label.rstrip('/')}/{pid}.{language}"


def parse_legacy_other_results_table(
    lines: list[str], start: int, end: int
) -> dict[int, dict[str, str]]:
    if start + 1 >= end:
        return {}
    header_cells = readme_tables.split_table_row(lines[start + 1])
    if not header_cells:
        return {}
    if header_cells[0] == "ID":
        suite_labels = header_cells[1:]
    else:
        suite_labels = header_cells
    rows: dict[int, dict[str, str]] = {}
    for i in range(start + 2, end - 1):
        cells = readme_tables.split_table_row(lines[i])
        if not cells:
            continue
        pid = parse_id_cell(cells[0])
        if pid is None:
            continue
        row_values = rows.setdefault(pid, {})
        for label, cell in zip(suite_labels, cells[1:]):
            cell = cell.strip()
            if not label or not cell:
                continue
            row_values[other_result_path_for_suite(pid, label, "py")] = cell
    return rows


def parse_other_results_table(lines: list[str]) -> dict[int, dict[str, str]]:
    try:
        start, end = other_results_block_span(lines)
    except RuntimeError:
        return {}
    if start >= end:
        return {}
    if lines[start].strip() == "|===":
        return parse_legacy_other_results_table(lines, start, end)

    rows: dict[int, dict[str, str]] = {}
    current_pid: int | None = None
    group_re = re.compile(r"^\*\s+(\d+)\s*$")
    entry_re = re.compile(r"^\*\*\s+(?:link:)?([^\[]+)\[[^\]]*\]:\s*(.*)$")
    for i in range(start, end):
        stripped = lines[i].strip()
        group_match = group_re.match(stripped)
        if group_match:
            current_pid = int(group_match.group(1))
            rows.setdefault(current_pid, {})
            continue
        entry_match = entry_re.match(stripped)
        if not entry_match:
            continue
        path_text = entry_match.group(1).strip()
        cell = entry_match.group(2).strip()
        if not path_text or not cell:
            continue
        path_pid = parse_solver_id(Path(path_text))
        pid = current_pid if current_pid is not None else path_pid
        if pid is None:
            continue
        rows.setdefault(pid, {})[path_text] = cell
    return rows


def parse_benchmark_value(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    if value.startswith('"'):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        return parsed if isinstance(parsed, str) else str(parsed)
    return value


def parse_benchmark_yaml(lines: list[str]) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    current_pid: int | None = None
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, sep, value = stripped.partition(":")
        if not sep:
            continue
        key = key.strip()
        if line == line.lstrip():
            if key.isdigit():
                current_pid = int(key)
                rows.setdefault(current_pid, {})
                continue
            path_text = key
            pid = parse_solver_id(Path(path_text))
        else:
            if current_pid is None:
                continue
            path_text = key
            pid = current_pid
        cell = parse_benchmark_value(value)
        if not path_text or not cell:
            continue
        if pid is None:
            continue
        rows.setdefault(pid, {})[path_text] = cell
    return rows


def load_benchmark_results(
    fallback_lines: list[str] | None = None,
) -> dict[int, dict[str, str]]:
    if BENCHMARK_PATH.exists():
        rows = parse_benchmark_yaml(BENCHMARK_PATH.read_text().splitlines())
        if rows or fallback_lines is None:
            return rows
    if fallback_lines is None:
        return {}
    return parse_other_results_table(fallback_lines)


def format_benchmark_value(value: str) -> str:
    value = value.strip()
    if looks_numeric(value):
        return value
    return json.dumps(value)


def format_benchmark_yaml(rows: dict[int, dict[str, str]]) -> list[str]:
    yaml_lines: list[str] = []
    for pid in sorted(rows):
        row_values = {
            path: cell.strip()
            for path, cell in rows[pid].items()
            if path.strip() and cell.strip()
        }
        if not row_values:
            continue
        yaml_lines.append(f"{pid}:")
        for path in sorted(row_values, key=other_result_path_sort_key):
            yaml_lines.append(
                f"  {path}: {format_benchmark_value(row_values[path])}"
            )
    return yaml_lines


def write_benchmark_results(rows: dict[int, dict[str, str]]) -> None:
    BENCHMARK_PATH.write_text("\n".join(format_benchmark_yaml(rows)) + "\n")


def format_other_results_list(rows: dict[int, dict[str, str]]) -> list[str]:
    list_lines: list[str] = []
    for pid in sorted(rows):
        row_values = {
            path: cell.strip()
            for path, cell in rows[pid].items()
            if path.strip() and cell.strip()
        }
        if not row_values:
            continue
        list_lines.append(f"* {pid}")
        for path in sorted(row_values, key=other_result_path_sort_key):
            list_lines.append(f"** link:{path}[]: {row_values[path]}")
    return list_lines


def replace_other_results_table(
    lines: list[str], rows: dict[int, dict[str, str]]
) -> None:
    start, end = other_results_block_span(lines)
    lines[start:end] = format_other_results_list(rows)


def other_result_language(path_text: str) -> str:
    return detect_language(Path(path_text)) or ""


def other_result_path_sort_key(path_text: str) -> tuple[int, str, str]:
    language = other_result_language(path_text)
    if language == "py":
        return (0, "", path_text)
    return (1, language, path_text)


def result_suite_label(res: Result) -> str:
    if res.source_path is not None:
        return relative_readme_path(res.source_path.parent)
    return "solvers"


def other_result_identity(res: Result) -> str | None:
    if not isinstance(res.puzzle_id, int) or res.source_path is None:
        return None
    return relative_readme_path(res.source_path)


def merge_other_result_cells(existing: str | None, incoming: str) -> str:
    if existing and looks_numeric(existing) and looks_numeric(incoming):
        return f"{min(float(existing), float(incoming)):.3f}"
    return incoming


def other_result_cell_for_error(message: str, output: str | None) -> str:
    if message.startswith("expected ") and output:
        return normalize_error_cell(f"{message}, got {output}")
    return normalize_error_cell(message)


def other_result_cell(res: Result) -> str:
    runtime_cell = comparison_runtime_cell(res)
    if runtime_cell is not None:
        return runtime_cell
    return other_result_cell_for_error(res.message, res.output)


def other_result_cell_from_row(
    time_cell: str, output_cell: str, error_cell: str
) -> str:
    if error_cell and not is_missing_reference_error_cell(error_cell):
        message = error_cell
        if message.startswith("error: "):
            message = message[len("error: ") :]
        return other_result_cell_for_error(message, output_cell)
    if time_cell:
        return time_cell
    if error_cell:
        return normalize_error_cell(error_cell)
    return ""


def primary_row_other_result_entry(pid: TestId, row: str) -> tuple[str, str] | None:
    if not isinstance(pid, int):
        return None
    cells = trim_trailing_empty_cells(readme_tables.split_table_row(row))
    normalized = normalize_row_fields(pid, cells)
    if normalized is None:
        return None
    (
        id_cell,
        _statement_cell,
        time_cell,
        output_cell,
        error_cell,
    ) = normalized
    path_text = link_target_from_cell(id_cell)
    if path_text is None:
        candidate = SOLVERS_DIR / id_cell.strip()
        if not candidate.exists():
            return None
        path_text = relative_readme_path(candidate)
    if other_result_language(path_text) != "py":
        return None
    cell = other_result_cell_from_row(time_cell, output_cell, error_cell)
    if not cell:
        return None
    return path_text, cell


def upsert_benchmark_results(
    results: list[Result],
    row_map: dict[tuple[TestId, str], str] | None = None,
    fallback_lines: list[str] | None = None,
) -> dict[int, dict[str, str]]:
    rows = load_benchmark_results(fallback_lines)

    if row_map is not None:
        for (pid, language), row in row_map.items():
            if language != "py":
                continue
            entry = primary_row_other_result_entry(pid, row)
            if entry is None:
                continue
            path_text, cell = entry
            rows.setdefault(pid, {})[path_text] = cell

    result_cells: dict[tuple[int, str], str] = {}
    for res in results:
        path_text = other_result_identity(res)
        if path_text is None or not isinstance(res.puzzle_id, int):
            continue
        key = (res.puzzle_id, path_text)
        cell = other_result_cell(res)
        result_cells[key] = merge_other_result_cells(result_cells.get(key), cell)

    for (pid, path_text), cell in result_cells.items():
        rows.setdefault(pid, {})[path_text] = cell

    write_benchmark_results(rows)
    return rows


def update_readme(results: list[Result]) -> None:
    readme_path = ROOT / "README.adoc"
    lines = readme_path.read_text().splitlines()
    start, end = readme_tables.find_table_block(lines, "// RESULTS TABLE")

    header_line = "| ID | Explanation | Runtime (s) | Output | Error"
    row_re = re.compile(r"^\|\s+link:([^\[]+)\[")
    plain_re = re.compile(r"^\|\s+(\d+)\.py\s+\|")
    result_map: dict[tuple[TestId, str], str] = {}
    row_map: dict[tuple[TestId, str], str] = {}

    for res in results:
        if is_primary_python_result(res):
            result_map[result_key(res)] = format_row(res)

    for i in range(start + 1, end):
        line = lines[i]
        match = row_re.match(line)
        if match:
            link_target = match.group(1)
            path = Path(link_target)
            if not is_primary_python_solver_path(path):
                continue
            pid = test_id_from_link_target(link_target)
            if pid is None:
                id_match = re.search(r"\[(\d+)\]", line)
                if not id_match:
                    continue
                pid = int(id_match.group(1))
            language = detect_language(path) or ""
            cells = trim_trailing_empty_cells(readme_tables.split_table_row(line))
            normalized = normalize_row_fields(pid, cells)
            if normalized is None:
                continue
            (
                id_cell,
                statement_cell,
                time_cell,
                output_cell,
                error_cell,
            ) = normalized
            row_map[(pid, language)] = format_row_fields(
                id_cell,
                statement_cell,
                time_cell,
                output_cell,
                error_cell,
            )
            continue
        plain_match = plain_re.match(line)
        if not plain_match:
            continue
        pid_text = plain_match.group(1)
        if not pid_text:
            continue
        pid = int(pid_text)
        cells = trim_trailing_empty_cells(readme_tables.split_table_row(line))
        normalized = normalize_row_fields(pid, cells)
        if normalized is None:
            continue
        (
            id_cell,
            statement_cell,
            time_cell,
            output_cell,
            error_cell,
        ) = normalized
        row_map[(pid, "py")] = format_row_fields(
            id_cell,
            statement_cell,
            time_cell,
            output_cell,
            error_cell,
        )

    for key, row in result_map.items():
        row_map[key] = row

    sorted_rows = [row_map[key] for key in sorted(row_map, key=row_sort_key)]
    lines[start + 1 : end] = [header_line, *sorted_rows]

    benchmark_rows = upsert_benchmark_results(results, row_map, lines)
    replace_other_results_table(lines, benchmark_rows)

    readme_path.write_text("\n".join(lines) + "\n")


def update_readme_solver_sets(results: list[Result]) -> None:
    readme_path = ROOT / "README.adoc"
    lines = readme_path.read_text().splitlines()
    row_map = parse_primary_python_row_map(lines)
    benchmark_rows = upsert_benchmark_results(results, row_map, lines)
    replace_other_results_table(lines, benchmark_rows)
    readme_path.write_text("\n".join(lines) + "\n")


def update_readme_links() -> None:
    readme_path = ROOT / "README.adoc"
    lines = readme_path.read_text().splitlines()
    start, end = readme_tables.find_table_block(lines, "// RESULTS TABLE")

    header_line = "| ID | Explanation | Runtime (s) | Output | Error"
    row_re = re.compile(r"^\|\s+link:([^\[]+)\[")
    plain_re = re.compile(r"^\|\s+(\d+)\.py\s+\|")
    row_map: dict[tuple[TestId, str], str] = {}

    for i in range(start + 1, end):
        line = lines[i]
        match = row_re.match(line)
        if match:
            link_target = match.group(1)
            path = Path(link_target)
            if not is_primary_python_solver_path(path):
                continue
            pid = test_id_from_link_target(link_target)
            if pid is None:
                continue
            language = detect_language(path) or ""
        else:
            plain_match = plain_re.match(line)
            if not plain_match:
                continue
            pid_text = plain_match.group(1)
            if not pid_text:
                continue
            pid = int(pid_text)
            language = "py"

        cells = trim_trailing_empty_cells(readme_tables.split_table_row(line))
        normalized = normalize_row_fields(pid, cells)
        if normalized is None:
            continue
        (
            id_cell,
            statement_cell,
            time_cell,
            output_cell,
            error_cell,
        ) = normalized
        row_map[(pid, language)] = format_row_fields(
            id_cell,
            statement_cell,
            time_cell,
            output_cell,
            error_cell,
        )

    sorted_rows = [row_map[key] for key in sorted(row_map, key=row_sort_key)]
    lines[start + 1 : end] = [header_line, *sorted_rows]
    readme_path.write_text("\n".join(lines) + "\n")


def update_readme_not_found() -> None:
    reference_ids = set(load_reference_answers())
    statement_ids = load_statement_ids()
    known_ids = reference_ids | statement_ids
    solver_entries = collect_solver_entries()
    solver_ids = {pid for pid, _language in solver_entries}
    readme_path = ROOT / "README.adoc"
    lines = readme_path.read_text().splitlines()
    start, end = readme_tables.find_table_block(lines, "// RESULTS TABLE")
    existing_other_rows = load_benchmark_results(lines)
    existing_other_paths = {
        path
        for path_values in existing_other_rows.values()
        for path in path_values
    }

    row_re = re.compile(r"^\|\s+link:([^\[]+)\[")
    plain_re = re.compile(r"^\|\s+(\d+)\.py\s+\|")
    row_map: dict[tuple[TestId, str], str] = {}
    result_map: dict[tuple[TestId, str], str] = {}
    other_results: list[Result] = []
    seen_ids: set[int] = set()

    for i in range(start + 1, end):
        line = lines[i]
        match = row_re.match(line)
        link_target = None
        if match:
            link_target = match.group(1)
            path = Path(link_target)
            if not is_primary_python_solver_path(path):
                continue
            pid = test_id_from_link_target(link_target)
            if pid is None:
                continue
            language = detect_language(path) or ""
        else:
            plain_match = plain_re.match(line)
            if not plain_match:
                continue
            pid_text = plain_match.group(1)
            if not pid_text:
                continue
            pid = int(pid_text)
            language = "py"

        if isinstance(pid, int):
            seen_ids.add(pid)
        cells = trim_trailing_empty_cells(readme_tables.split_table_row(line))
        normalized = normalize_row_fields(pid, cells)
        if normalized is None:
            continue
        (
            id_cell,
            statement_cell,
            time_cell,
            output_cell,
            error_cell,
        ) = normalized
        row_map[(pid, language)] = format_row_fields(
            id_cell,
            statement_cell,
            time_cell,
            output_cell,
            error_cell,
        )
        if link_target:
            path = ROOT / link_target
        else:
            path = SOLVERS_DIR / f"{pid}.py"
        if path.exists():
            continue
        res = Result(
            puzzle_id=pid,
            correct=False,
            elapsed=None,
            output=None,
            message="solver not found",
            language=language,
            source_path=None,
        )
        result_map[result_key(res)] = format_row(res)

    for key, path in solver_entries.items():
        pid, language = key
        if language == "py":
            if key in row_map:
                continue
        elif relative_readme_path(path) in existing_other_paths:
            continue
        res = Result(
            puzzle_id=pid,
            correct=False,
            elapsed=None,
            output=None,
            message="untested",
            language=language,
            source_path=path,
        )
        if language == "py":
            result_map[result_key(res)] = format_row(res)
        else:
            other_results.append(res)

    for name, target in EXPLICIT_ONLY_TESTS.items():
        key = (name, target.language)
        if target.language == "py":
            if key in row_map:
                continue
        elif relative_readme_path(target.path) in existing_other_paths:
            continue
        res = Result(
            puzzle_id=name,
            correct=False,
            elapsed=None,
            output=None,
            message="untested",
            language=target.language,
            source_path=target.path,
            reference_answer_checked=target.checks_reference_answer,
        )
        if target.language == "py":
            result_map[result_key(res)] = format_row(res)
        else:
            other_results.append(res)

    for pid in sorted(known_ids):
        if pid in seen_ids or pid in solver_ids:
            continue
        res = Result(
            puzzle_id=pid,
            correct=False,
            elapsed=None,
            output=None,
            message="solver not found",
            language="py",
            source_path=None,
        )
        result_map[result_key(res)] = format_row(res)

    for key, row in result_map.items():
        row_map[key] = row

    sorted_rows = [row_map[key] for key in sorted(row_map, key=row_sort_key)]
    lines[start + 1 : end] = [
        "| ID | Explanation | Runtime (s) | Output | Error",
        *sorted_rows,
    ]

    benchmark_rows = upsert_benchmark_results(other_results, row_map, lines)
    replace_other_results_table(lines, benchmark_rows)

    readme_path.write_text("\n".join(lines) + "\n")


def parse_lang_filter(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    allowed = {"py", "c", "cpp", "lean"}
    selected: set[str] = set()
    for value in values:
        for token in value.split(","):
            token = token.strip().lower()
            if not token:
                continue
            if token not in allowed:
                raise ValueError(f"invalid language: {token}")
            selected.add(token)
    return selected


def run_solver_set_target(
    target: SolverTarget,
    reference: dict[int, str],
    timeout: float | None,
) -> Result:
    pid = target.puzzle_id
    expected = reference.get(pid) if isinstance(pid, int) else None
    missing_reference = expected is None

    print(f"[{pid}] running {target.path} ({target.suite})")
    rc, stdout, stderr, elapsed, timed_out = run_solver(
        target.path, timeout, target.language
    )
    source_path = source_from_target(target.path, target.language)
    if timed_out:
        limit = timeout if timeout is not None else elapsed
        print(f"[{pid}] timed out after {limit:.3f}s", file=sys.stderr)
        return Result(
            pid,
            correct=False,
            elapsed=None,
            output=None,
            message=f"timed out after {limit:.3f}s",
            language=target.language,
            source_path=source_path,
        )
    if rc != 0:
        if stderr.strip():
            print(stderr.rstrip(), file=sys.stderr)
        print(f"[{pid}] failed (exit {rc})", file=sys.stderr)
        return Result(
            pid,
            correct=False,
            elapsed=elapsed,
            output=None,
            message=f"failed (exit {rc})",
            language=target.language,
            source_path=source_path,
        )

    display_output = format_stdout_for_display(stdout)
    if missing_reference:
        print(f"[{pid}] missing reference answer", file=sys.stderr)
        return Result(
            pid,
            correct=False,
            elapsed=elapsed,
            output=display_output,
            message=MISSING_REFERENCE_MESSAGE,
            language=target.language,
            source_path=source_path,
        )
    if stdout_matches_expected(stdout, expected):
        print(f"[{pid}] ok ({elapsed:.3f}s)")
        return Result(
            pid,
            correct=True,
            elapsed=elapsed,
            output=expected,
            message="ok",
            language=target.language,
            source_path=source_path,
        )
    msg = f"expected {expected}"
    print(
        f"[{pid}] wrong answer: {wrong_answer_detail(display_output, expected)}",
        file=sys.stderr,
    )
    return Result(
        pid,
        correct=False,
        elapsed=elapsed,
        output=display_output,
        message=msg,
        language=target.language,
        source_path=source_path,
    )


def suite_result_cells(results: list[Result]) -> dict[tuple[int, str], str]:
    cells: dict[tuple[int, str], str] = {}
    for res in results:
        if not isinstance(res.puzzle_id, int):
            continue
        key = (res.puzzle_id, result_suite_label(res))
        cell = comparison_runtime_cell(res)
        if cell is not None:
            existing = cells.get(key)
            if existing:
                cell = f"{min(float(existing), float(cell)):.3f}"
            cells[key] = cell
        elif key not in cells:
            cells[key] = ""
    return cells


def print_solver_set_table(
    solver_sets: list[SolverSet],
    target_ids: list[int],
    results: list[Result],
) -> None:
    labels = [solver_set.label for solver_set in solver_sets]
    cells = suite_result_cells(results)
    print("\n|===")
    print("| ID | " + " | ".join(labels))
    for pid in target_ids:
        row = [str(pid)]
        row.extend(cells.get((pid, label), "") for label in labels)
        print("| " + " | ".join(row))
    print("|===")


def run_solver_set_mode(
    args: argparse.Namespace,
    solver_sets: list[SolverSet],
    reference: dict[int, str],
    lang_filter: set[str] | None,
    ids_values: list[str],
) -> None:
    if args.uncommitted:
        print("--uncommitted is not supported with --set", file=sys.stderr)
        sys.exit(2)

    targets_by_set = {
        solver_set.label: collect_solver_set_targets(solver_set, lang_filter)
        for solver_set in solver_sets
    }
    if ids_values:
        try:
            target_ids = expand_solver_set_ids(ids_values)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(2)
    else:
        target_ids = sorted(
            {
                pid
                for targets_by_id in targets_by_set.values()
                for pid in targets_by_id
            }
        )
    if not target_ids:
        print("No solvers found.", file=sys.stderr)
        sys.exit(1)

    results: list[Result] = []
    for pid in target_ids:
        for solver_set in solver_sets:
            targets = targets_by_set[solver_set.label].get(pid, [])
            for target in targets:
                results.append(run_solver_set_target(target, reference, args.timeout))

    if not results:
        print("No implemented solvers found for selected IDs.", file=sys.stderr)
        sys.exit(1)

    total_run = len(results)
    passed = sum(r.correct for r in results)
    print(f"\nPassed {passed}/{total_run} implemented tests.")
    print_solver_set_table(solver_sets, target_ids, results)

    if args.autoupdate:
        update_readme_solver_sets(results)
        try:
            summary.autoupdate_readme()
        except (OSError, ValueError) as exc:
            print(f"summary update failed: {exc}", file=sys.stderr)
            sys.exit(2)

    if any(
        not res.correct and res.message != MISSING_REFERENCE_MESSAGE
        for res in results
    ):
        sys.exit(1)


def main() -> None:
    args = parse_args()
    if args.autoupdate_links:
        update_readme_links()
        return
    if args.autoupdate_not_found:
        update_readme_not_found()
        return
    reference = load_reference_answers()
    try:
        explicit_langs = parse_lang_filter(args.lang)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
    lang_filter = explicit_langs
    try:
        solver_sets = parse_solver_sets(args.sets)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
    ids_values = list(args.ids or [])
    if ids_values:
        ids_values, lang_hints = extract_lang_hints(ids_values)
        if lang_hints:
            if lang_filter is None:
                lang_filter = set()
            lang_filter |= lang_hints
    if solver_sets:
        run_solver_set_mode(args, solver_sets, reference, lang_filter, ids_values)
        return
    solver_targets = collect_solver_targets(lang_filter)
    explicit_only_targets = collect_explicit_only_targets(lang_filter)

    path_overrides: dict[TestId, list[Path]] = {}
    if args.uncommitted:
        try:
            target_ids = collect_uncommitted_solvers()
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(2)
    elif ids_values:
        try:
            target_ids, path_overrides = expand_ids(ids_values)
            target_ids = sort_test_ids(target_ids)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(2)
    else:
        target_ids = sorted(solver_targets)
    if not target_ids:
        print("No solvers found.", file=sys.stderr)
        sys.exit(1)

    lint_paths: set[Path] = set()
    for pid in target_ids:
        override_paths = path_overrides.get(pid)
        targets = targets_for_id(pid, solver_targets, explicit_only_targets)
        if override_paths:
            for path in override_paths:
                language = detect_language(path)
                if language is None:
                    continue
                if language == "lean":
                    continue
                lint_paths.add(source_from_target(path, language))
        else:
            for target in targets:
                if target.language == "lean":
                    continue
                lint_paths.add(source_from_target(target.path, target.language))

    violations = lint.lint_paths(sorted(path for path in lint_paths if path.exists()))
    if violations:
        for line in lint.format_violations(violations, root=ROOT):
            print(line, file=sys.stderr)
    critical_lint_violations = lint.critical_violations(violations)
    if critical_lint_violations:
        lint_results: list[Result] = []
        for violation in critical_lint_violations:
            language = detect_language(violation.path)
            lint_results.append(
                Result(
                    violation.puzzle_id,
                    correct=False,
                    elapsed=None,
                    output=None,
                    message="critical lint failed",
                    language=language,
                    source_path=violation.path,
                )
            )
        print("\n|===")
        print("| ID | Explanation | Runtime (s) | Output | Error")
        for res in sorted(lint_results, key=result_sort_key):
            print(format_row(res))
        print("|===")
        if args.autoupdate:
            update_readme(lint_results)
            try:
                summary.autoupdate_readme()
            except (OSError, ValueError) as exc:
                print(f"summary update failed: {exc}", file=sys.stderr)
                sys.exit(2)
        sys.exit(1)

    results: list[Result] = []
    for pid in target_ids:
        override_paths = path_overrides.get(pid)
        targets: list[SolverTarget] = []
        if override_paths:
            for path in override_paths:
                language = detect_language(path)
                if language is None:
                    print(f"invalid solver path: {path}", file=sys.stderr)
                    sys.exit(2)
                if lang_filter and language not in lang_filter:
                    print(
                        f"solver path {path} does not match --lang filter",
                        file=sys.stderr,
                    )
                    sys.exit(2)
                targets.append(SolverTarget(pid, path, language))
        else:
            targets = targets_for_id(pid, solver_targets, explicit_only_targets)
        if not targets:
            results.append(
                Result(
                    pid,
                    correct=False,
                    elapsed=None,
                    output=None,
                    message="solver not found",
                    language=None,
                    source_path=None,
                )
            )
            print(f"[{pid}] skipped: solver not found", file=sys.stderr)
            continue

        for target in targets:
            expected = None
            if target.checks_reference_answer and isinstance(pid, int):
                expected = reference.get(pid)
            missing_reference = target.checks_reference_answer and expected is None
            label = f"{target.language}" if target.language else "unknown"
            print(f"[{pid}] running {target.path} ({label})")
            rc, stdout, stderr, elapsed, timed_out = run_solver(
                target.path, args.timeout, target.language
            )
            if timed_out:
                limit = args.timeout if args.timeout is not None else elapsed
                results.append(
                    Result(
                        pid,
                        correct=False,
                        elapsed=None,
                        output=None,
                        message=f"timed out after {limit:.3f}s",
                        language=target.language,
                        source_path=source_from_target(target.path, target.language),
                        reference_answer_checked=target.checks_reference_answer,
                    )
                )
                print(f"[{pid}] timed out after {limit:.3f}s", file=sys.stderr)
                continue

            if rc != 0:
                if stderr.strip():
                    print(stderr.rstrip(), file=sys.stderr)
                results.append(
                    Result(
                        pid,
                        correct=False,
                        elapsed=elapsed,
                        output=None,
                        message=f"failed (exit {rc})",
                        language=target.language,
                        source_path=source_from_target(target.path, target.language),
                        reference_answer_checked=target.checks_reference_answer,
                    )
                )
                print(f"[{pid}] failed (exit {rc})", file=sys.stderr)
                continue

            display_output = format_stdout_for_display(stdout)
            if not target.checks_reference_answer:
                results.append(
                    Result(
                        pid,
                        correct=True,
                        elapsed=elapsed,
                        output=None,
                        message="output ignored",
                        language=target.language,
                        source_path=source_from_target(target.path, target.language),
                        reference_answer_checked=False,
                    )
                )
                print(f"[{pid}] ok ({elapsed:.3f}s; output ignored)")
                continue
            if missing_reference:
                results.append(
                    Result(
                        pid,
                        correct=False,
                        elapsed=elapsed,
                        output=display_output,
                        message=MISSING_REFERENCE_MESSAGE,
                        language=target.language,
                        source_path=source_from_target(target.path, target.language),
                        reference_answer_checked=target.checks_reference_answer,
                    )
                )
                print(f"[{pid}] missing reference answer", file=sys.stderr)
                continue

            if stdout_matches_expected(stdout, expected):
                results.append(
                    Result(
                        pid,
                        correct=True,
                        elapsed=elapsed,
                        output=expected,
                        message="ok",
                        language=target.language,
                        source_path=source_from_target(target.path, target.language),
                        reference_answer_checked=target.checks_reference_answer,
                    )
                )
                print(f"[{pid}] ok ({elapsed:.3f}s)")
            else:
                msg = f"expected {expected}"
                results.append(
                    Result(
                        pid,
                        correct=False,
                        elapsed=elapsed,
                        output=display_output,
                        message=msg,
                        language=target.language,
                        source_path=source_from_target(target.path, target.language),
                        reference_answer_checked=target.checks_reference_answer,
                    )
                )
                print(
                    f"[{pid}] wrong answer: {wrong_answer_detail(display_output, expected)}",
                    file=sys.stderr,
                )

    total_run = len(results)
    passed = sum(r.correct for r in results)
    print(f"\nPassed {passed}/{total_run} tests.")

    py_results = [res for res in results if is_primary_python_result(res)]
    other_results = [res for res in results if not is_primary_python_result(res)]

    if py_results:
        print("\n|===")
        print("| ID | Explanation | Runtime (s) | Output | Error")
        for res in sorted(py_results, key=result_sort_key):
            print(format_row(res))
        print("|===")

    if other_results:
        py_runtime: dict[TestId, float] = {}
        for res in py_results:
            if res.elapsed is not None:
                py_runtime[res.puzzle_id] = res.elapsed
        print("\n|===")
        print("| ID | Runtime (s) | Python runtime x | Output | Error")
        for res in sorted(other_results, key=result_sort_key):
            ratio = ""
            py_time = py_runtime.get(res.puzzle_id)
            if res.elapsed is not None and py_time:
                ratio = f"{(res.elapsed / py_time):.2f}"
            row = format_row_other(res)
            if row.startswith("|"):
                row = row.lstrip("| ").rstrip()
            row = row.rstrip()
            cells = [cell.strip() for cell in row.split("|")]
            if len(cells) >= 4:
                id_cell, time_cell, output_cell, error_cell = cells[:4]
                row = (
                    f"| {id_cell} | {time_cell} | {ratio} | "
                    f"{output_cell} | {error_cell}"
                ).rstrip()
            else:
                row = f"| {row} | {ratio}".rstrip()
            print(row)
        print("|===")

    if args.autoupdate:
        update_readme(results)
        try:
            summary.autoupdate_readme()
        except (OSError, ValueError) as exc:
            print(f"summary update failed: {exc}", file=sys.stderr)
            sys.exit(2)

    if any(not res.correct for res in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
