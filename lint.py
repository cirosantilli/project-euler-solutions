#!/usr/bin/env python3
from __future__ import annotations

import io
import argparse
import re
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parent
README_PATH = ROOT / "README.adoc"
SOLUTIONS_PATH = ROOT / "data/projecteuler-solutions/Solutions.md"
SOLVERS_DIR = ROOT / "solvers"
LEAN_SOLVERS_DIR = ROOT / "ProjectEulerSolutions"
LEAN_EQUIV_DIR = LEAN_SOLVERS_DIR / "Equiv"
LEAN_TERMINATION_DIR = LEAN_SOLVERS_DIR / "Termination"
VALID_PYTHON_SHEBANG = "#!/usr/bin/env python"
SOURCE_EXTENSIONS = (".py", ".c", ".cpp")
FORMAL_STATUS_MARKER = "// FORMAL STATUS TABLE"

LINE_RE = re.compile(r"^(\d+)\.\s+(.*)$")
LEAN_SOLUTION_STEM_RE = re.compile(r"^P(\d+)$")
LEAN_EQUIV_STEM_RE = re.compile(r"^P(\d+)$")
FORBIDDEN_TOKENS = ("Solutions.md",)
FORBIDDEN_LEAN_PATTERNS = (
    (r"\baxiom\b", "axiom"),
    (r"\badmit\b", "admit"),
    (r"\bnoncomputable\b", "noncomputable"),
    (r"\bpartial\b", "partial"),
    (r"\bsorry\b", "sorry"),
    (r"\bunsafe\b", "unsafe"),
)
FORBIDDEN_LEAN_DEF_ARG_RE = re.compile(r"\bdef\s+\w+[^\n]*\b_[A-Za-z0-9_]*\b")
ViolationSeverity = Literal["critical", "non-critical"]


@dataclass(frozen=True)
class Violation:
    kind: str
    puzzle_id: int
    path: Path
    answer: str
    context: list[tuple[int, str]]
    severity: ViolationSeverity = "critical"


def load_reference_answers() -> dict[int, str]:
    solutions: dict[int, str] = {}
    with SOLUTIONS_PATH.open() as fh:
        for line in fh:
            line = line.strip()
            match = LINE_RE.match(line)
            if not match:
                continue
            pid = int(match.group(1))
            solutions[pid] = match.group(2).strip()
    return solutions


def parse_solver_id(path: Path) -> int | None:
    stem = path.stem
    if stem.isdigit():
        return int(stem)
    match = LEAN_SOLUTION_STEM_RE.match(stem)
    if match:
        return int(match.group(1))
    match = LEAN_EQUIV_STEM_RE.match(stem)
    if match:
        return int(match.group(1))
    if "_" in stem:
        prefix = stem.split("_", 1)[0]
        if prefix.isdigit():
            return int(prefix)
    return None


def iter_sources_in(source_dir: Path, extensions: tuple[str, ...]) -> list[Path]:
    if not source_dir.is_absolute():
        source_dir = ROOT / source_dir
    sources: list[Path] = []
    for ext in extensions:
        sources.extend(source_dir.glob(f"*{ext}"))
    return sorted(sources)


def iter_solver_sources() -> list[Path]:
    return iter_sources_in(SOLVERS_DIR, SOURCE_EXTENSIONS)


def iter_source_set_sources(source_set: Path) -> list[Path]:
    return iter_sources_in(source_set, SOURCE_EXTENSIONS + (".lean",))


def should_scan_answer(answer: str) -> bool:
    return len(answer) > 1


def python_comment_or_string_hits(text: str, answer: str) -> list[int]:
    hits: list[int] = []
    reader = io.StringIO(text).readline
    try:
        for tok in tokenize.generate_tokens(reader):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                if answer in tok.string:
                    hits.append(tok.start[0])
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return hits
    return hits


def python_line_hits(text: str, answer: str) -> list[int]:
    hits: list[int] = []
    answer_re = re.escape(answer)
    answer_literal = rf"(?:['\"]{answer_re}['\"]|(?<![\w.]){answer_re}(?![\w.]))"
    action_re = re.compile(rf"\b(?:return|print)\b.*{answer_literal}")
    assignment_re = re.compile(rf"(?<![!<>=])=\s*{answer_literal}")
    comparison_re = re.compile(
        rf"(?:==|!=)\s*{answer_literal}|{answer_literal}\s*(?:==|!=)"
    )
    for idx, line in enumerate(text.splitlines(), 1):
        code = line.split("#", 1)[0]
        if answer not in code:
            continue
        if (
            action_re.search(code)
            or assignment_re.search(code)
            or comparison_re.search(code)
        ):
            hits.append(idx)
    return hits


def c_comment_hits(text: str, answer: str) -> list[int]:
    hits: list[int] = []
    in_block = False
    for idx, line in enumerate(text.splitlines(), 1):
        if in_block:
            if answer in line:
                hits.append(idx)
            if "*/" in line:
                in_block = False
            continue
        if "/*" in line:
            in_block = True
            if answer in line:
                hits.append(idx)
            if "*/" in line:
                in_block = False
            continue
        if "//" in line:
            if answer in line.split("//", 1)[1]:
                hits.append(idx)
    return hits


def c_line_hits(text: str, answer: str) -> list[int]:
    hits: list[int] = []
    in_block = False
    answer_re = re.escape(answer)
    suffix_re = r"[uUlL]*" if answer.isdigit() else ""
    literal_re = re.compile(rf"(?<![\w.]){answer_re}{suffix_re}(?![\w.])")
    quoted_re = re.compile(rf"['\"]{answer_re}['\"]")
    for idx, line in enumerate(text.splitlines(), 1):
        if in_block:
            if "*/" in line:
                in_block = False
            continue
        if "/*" in line:
            if "*/" not in line:
                in_block = True
            continue
        if "//" in line:
            line = line.split("//", 1)[0]
        normalized_line = re.sub(r"(?<=\d)'(?=\d)", "", line)
        if answer not in normalized_line:
            continue
        if re.search(r"\breturn\b", normalized_line):
            hits.append(idx)
            continue
        if re.search(r"\bprintf\s*\(", normalized_line):
            hits.append(idx)
            continue
        if re.search(r"\bcout\b", normalized_line):
            hits.append(idx)
            continue
        if re.search(r"\bassert\s*\(", normalized_line):
            hits.append(idx)
            continue
        comparison_re = (
            rf"(?:==|!=)\s*(?:{quoted_re.pattern}|{literal_re.pattern})"
            rf"|(?:{quoted_re.pattern}|{literal_re.pattern})\s*(?:==|!=)"
        )
        if re.search(comparison_re, normalized_line):
            hits.append(idx)
            continue
        if re.search(
            rf"(?<![!<>=])=\s*(?:{quoted_re.pattern}|{literal_re.pattern})",
            normalized_line,
        ):
            hits.append(idx)
    return hits


def lean_code_without_comments(text: str) -> str:
    """Mask nested Lean comments while preserving line numbers."""
    masked = list(text)
    depth = 0
    index = 0
    while index < len(text):
        if text.startswith("/-", index):
            depth += 1
            masked[index:index + 2] = "  "
            index += 2
        elif depth and text.startswith("-/", index):
            depth -= 1
            masked[index:index + 2] = "  "
            index += 2
        elif depth:
            if text[index] != "\n":
                masked[index] = " "
            index += 1
        elif text.startswith("--", index):
            end = text.find("\n", index)
            if end == -1:
                end = len(text)
            masked[index:end] = " " * (end - index)
            index = end
        else:
            index += 1
    return "".join(masked)


def forbidden_hits(text: str, token: str) -> list[int]:
    return [idx for idx, line in enumerate(text.splitlines(), 1) if token in line]


def python_shebang_violation(
    path: Path, pid: int, lines: list[str]
) -> Violation | None:
    if path.resolve().parent != SOLVERS_DIR or path.suffix != ".py":
        return None
    first_line = lines[0] if lines else ""
    if first_line == VALID_PYTHON_SHEBANG:
        return None
    if first_line.startswith("#!"):
        message = f"must start with {VALID_PYTHON_SHEBANG!r}"
        context = [(1, first_line.rstrip())]
    else:
        message = f"missing shebang {VALID_PYTHON_SHEBANG!r}"
        context = [(1, first_line.rstrip())] if lines else []
    return Violation("shebang", pid, path, message, context, severity="non-critical")


def lint_paths(
    paths: list[Path],
    answers: dict[int, str] | None = None,
    scan_code_answers: bool = False,
) -> list[Violation]:
    if answers is None:
        answers = load_reference_answers()
    violations: list[Violation] = []
    for path in paths:
        pid = parse_solver_id(path)
        if pid is None:
            continue
        if path.suffix == ".lean":
            try:
                text = path.read_text()
            except OSError as exc:
                print(f"error: failed to read {path}: {exc}", file=sys.stderr)
                continue
            lines = text.splitlines()
            code = lean_code_without_comments(text)
            code_lines = code.splitlines()
            for pattern, token_name in FORBIDDEN_LEAN_PATTERNS:
                hits = [
                    idx for idx, line in enumerate(code_lines, 1) if re.search(pattern, line)
                ]
                if hits:
                    context = [
                        (line_no, lines[line_no - 1].rstrip()) for line_no in hits
                    ]
                    violations.append(
                        Violation(
                            "lean",
                            pid,
                            path,
                            f"contains forbidden token {token_name!r}",
                            context,
                        )
                    )
            def_arg_hits = [
                idx
                for idx, line in enumerate(code_lines, 1)
                if FORBIDDEN_LEAN_DEF_ARG_RE.search(line.split(":=", 1)[0])
            ]
            if def_arg_hits:
                context = [
                    (line_no, lines[line_no - 1].rstrip()) for line_no in def_arg_hits
                ]
                violations.append(
                    Violation(
                        "lean",
                        pid,
                        path,
                        "def arguments must not start with '_'",
                        context,
                    )
                )
            if path.parent == SOLVERS_DIR:
                main_parts = re.split(r"\bdef\s+main\b", code, maxsplit=1)
                main_body = main_parts[1] if len(main_parts) == 2 else ""
                if not (re.search(r"\bsolve\b", main_body) and
                        re.search(r"\bIO\.println\b", main_body)):
                    context: list[tuple[int, str]] = []
                    if lines:
                        context = [(len(lines), lines[-1].rstrip())]
                    violations.append(
                        Violation(
                            "lean",
                            pid,
                            path,
                            "main must call solve and print its result",
                            context,
                        )
                    )
            elif path.parent == LEAN_EQUIV_DIR and LEAN_EQUIV_STEM_RE.match(path.stem):
                normalized_text = " ".join(code.split())
                # Reference models and solver signatures may have different proof
                # arguments. Lean checks the equality and its domain; this check
                # only enforces the exported theorem's presence and result shape.
                match = re.search(r"\btheorem\s+equiv\b(.*?)\s:=", normalized_text)
                declaration = match.group(1) if match else ""
                if not (re.search(r"\s=\s", declaration) and
                        re.search(r"\bsolve\b", declaration)):
                    context: list[tuple[int, str]] = []
                    for idx, line in enumerate(lines, 1):
                        if "theorem equiv" in line:
                            context = [(idx, line.rstrip())]
                            break
                    violations.append(
                        Violation(
                            "lean",
                            pid,
                            path,
                            "missing required theorem declaration",
                            context,
                        )
                    )
            elif path.parent == LEAN_SOLVERS_DIR and LEAN_SOLUTION_STEM_RE.match(path.stem):
                theorem_hits = [
                    idx
                    for idx, line in enumerate(lines, 1)
                    if re.search(r"\btheorem\s+equiv\b", line)
                ]
                if theorem_hits:
                    context = [
                        (line_no, lines[line_no - 1].rstrip())
                        for line_no in theorem_hits
                    ]
                    violations.append(
                        Violation(
                            "lean",
                            pid,
                            path,
                            "theorem equiv belongs in Equiv/P<n>.lean",
                            context,
                        )
                    )
            continue
        if path.suffix not in (".py", ".c", ".cpp"):
            continue
        try:
            text = path.read_text()
        except OSError as exc:
            print(f"error: failed to read {path}: {exc}", file=sys.stderr)
            continue
        lines = text.splitlines()
        shebang_violation = python_shebang_violation(path, pid, lines)
        if shebang_violation is not None:
            violations.append(shebang_violation)
        answer = answers.get(pid)
        if answer and should_scan_answer(answer):
            if path.suffix == ".py":
                line_hits = python_comment_or_string_hits(text, answer)
                if scan_code_answers:
                    line_hits += python_line_hits(text, answer)
            else:
                line_hits = c_comment_hits(text, answer)
                line_hits += c_line_hits(text, answer)
            if line_hits:
                hit_lines = sorted(set(line_hits))
                context = [
                    (line_no, lines[line_no - 1].rstrip())
                    for line_no in hit_lines
                    if 0 < line_no <= len(lines)
                ]
                violations.append(Violation("answer", pid, path, answer, context))
        for token in FORBIDDEN_TOKENS:
            token_hits = forbidden_hits(text, token)
            if not token_hits:
                continue
            hit_lines = sorted(set(token_hits))
            context = [
                (line_no, lines[line_no - 1].rstrip())
                for line_no in hit_lines
                if 0 < line_no <= len(lines)
            ]
            violations.append(Violation("forbidden", pid, path, token, context))
    return violations


def critical_violations(violations: list[Violation]) -> list[Violation]:
    return [violation for violation in violations if violation.severity == "critical"]


def non_critical_violations(violations: list[Violation]) -> list[Violation]:
    return [violation for violation in violations if violation.severity != "critical"]


def _format_violation_details(
    violations: list[Violation], root: Path = ROOT
) -> list[str]:
    lines: list[str] = []
    sorted_violations = sorted(
        violations, key=lambda v: (v.puzzle_id, str(v.path), v.kind, v.answer)
    )
    for violation in sorted_violations:
        rel = violation.path
        try:
            rel = violation.path.relative_to(root)
        except ValueError:
            rel = violation.path
        if violation.kind == "forbidden":
            detail = f"contains forbidden token {violation.answer!r}"
        elif violation.kind in ("lean", "shebang"):
            detail = violation.answer
        else:
            detail = f"contains reference answer {violation.answer!r}"
        lines.append(f"- {violation.puzzle_id}: {rel} {detail}")
        for line_no, line in violation.context:
            lines.append(f"  line {line_no}: {line}")
    return lines


def format_violations(violations: list[Violation], root: Path = ROOT) -> list[str]:
    if not violations:
        return ["ok: no lint violations found in solver sources."]
    critical = critical_violations(violations)
    non_critical = non_critical_violations(violations)
    lines: list[str] = []
    if critical:
        lines.append("error: critical lint violations found in solver sources:")
        lines.extend(_format_violation_details(critical, root=root))
    if non_critical:
        if lines:
            lines.append("")
        lines.append("warning: non-critical lint violations found in solver sources:")
        lines.extend(_format_violation_details(non_critical, root=root))
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check solver sources for lint violations."
    )
    parser.add_argument(
        "-l",
        "--language",
        choices=("py", "c", "cpp", "lean"),
        help="Only lint a specific language.",
    )
    parser.add_argument(
        "--set",
        dest="source_set",
        type=Path,
        help=(
            "Source directory to lint when no explicit paths are provided "
            "(for example: solvers/eulersolve). Defaults to solvers."
        ),
    )
    parser.add_argument(
        "-A",
        "--autoupdate",
        action="store_true",
        help="Update the Python formalsw and Lean status lists in README.adoc.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Optional solver source paths to lint (defaults to all solvers).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.paths:
        paths = sorted(args.paths)
    else:
        paths = (
            iter_source_set_sources(args.source_set)
            if args.source_set is not None
            else iter_solver_sources()
        )
    if args.language == "lean":
        if args.paths:
            paths = [path for path in paths if path.suffix == ".lean"]
        elif args.source_set is None:
            paths = sorted(LEAN_SOLVERS_DIR.glob("*.lean"))
            paths.extend(sorted(LEAN_EQUIV_DIR.glob("*.lean")))
            paths.extend(sorted(LEAN_TERMINATION_DIR.glob("*.lean")))
            paths.extend(sorted(SOLVERS_DIR.glob("*.lean")))
        else:
            paths = [path for path in paths if path.suffix == ".lean"]
    elif args.language:
        lang_ext = f".{args.language}"
        paths = [path for path in paths if path.suffix == lang_ext]
    else:
        paths = [path for path in paths if path.suffix != ".lean"]
    if args.autoupdate:
        try:
            formal_status_lines = autoupdate_formal_status()
        except (OSError, ValueError) as exc:
            print(f"error: failed to update formal status table: {exc}", file=sys.stderr)
            return 2
        for line in formal_status_lines:
            print(line)
    violations = lint_paths(paths, scan_code_answers=args.source_set is not None)
    output_lines = format_violations(violations)
    if violations:
        for line in output_lines:
            print(line)
        return 1
    print(output_lines[0])
    return 0


def lean_problem_ids() -> list[int]:
    ids: set[int] = set()
    for directory in (LEAN_SOLVERS_DIR, LEAN_EQUIV_DIR, LEAN_TERMINATION_DIR):
        for path in directory.glob("P*.lean"):
            pid = parse_solver_id(path)
            if pid is not None:
                ids.add(pid)
    for path in SOLVERS_DIR.glob("*.lean"):
        pid = parse_solver_id(path)
        if pid is not None:
            ids.add(pid)
    return sorted(ids)


def lean_problem_paths(pid: int) -> list[Path]:
    paths = [
        SOLVERS_DIR / f"{pid}.lean",
        LEAN_SOLVERS_DIR / f"P{pid}.lean",
        LEAN_EQUIV_DIR / f"P{pid}.lean",
    ]
    termination_path = LEAN_TERMINATION_DIR / f"P{pid}.lean"
    if termination_path.exists():
        paths.append(termination_path)
    return paths


def compute_lean_status() -> tuple[dict[int, bool], int]:
    ids = lean_problem_ids()
    if not ids:
        return {}, 0
    max_pid = max(ids)
    solved_by_pid: dict[int, bool] = {}
    for pid in range(1, max_pid + 1):
        paths = lean_problem_paths(pid)
        if not all(path.exists() for path in paths):
            solved_by_pid[pid] = False
            continue
        solved_by_pid[pid] = not lint_paths(paths)
    return solved_by_pid, max_pid


def build_formal_status_lines(solved_by_pid: dict[int, bool], max_pid: int) -> list[str]:
    if max_pid == 0:
        return ["** no entries found"]
    raw_groups: list[tuple[int, int, int, int, list[int]]] = []
    for start in range(1, max_pid + 1, 10):
        end = min(start + 9, max_pid)
        missing = [
            pid for pid in range(start, end + 1) if not solved_by_pid.get(pid, False)
        ]
        total = end - start + 1
        solved = total - len(missing)
        raw_groups.append((start, end, solved, total, missing))

    output: list[str] = []
    pending_done_start: int | None = None
    pending_done_end: int | None = None

    def flush_done() -> None:
        nonlocal pending_done_start, pending_done_end
        if pending_done_start is not None and pending_done_end is not None:
            output.append(f"** **{pending_done_start}-{pending_done_end}**: done")
        pending_done_start = None
        pending_done_end = None

    for start, end, solved, total, missing in raw_groups:
        if solved == total:
            if pending_done_start is None:
                pending_done_start = start
            pending_done_end = end
            continue
        flush_done()
        line = f"** **{start}-{end}**: {solved}/{total}"
        if missing:
            line += " (TODO: " + ", ".join(str(pid) for pid in missing) + ")"
        output.append(line)
    flush_done()
    return output


def compute_formalsw_status() -> tuple[dict[int, bool], int]:
    # Track the proof work's scope, not every unannotated Python solution.
    sources = {
        int(path.stem): path
        for path in SOLVERS_DIR.glob("*.py")
        if path.stem.isdigit()
    }
    ids = {
        pid for pid, path in sources.items()
        if "@formalsw:" in path.read_text()
    }
    ids.update(
        int(path.name.removesuffix(".py.lean"))
        for path in SOLVERS_DIR.glob("*.py.lean")
        if path.name.removesuffix(".py.lean").isdigit()
    )
    max_pid = max(ids, default=0)
    solved_by_pid: dict[int, bool] = {}
    for pid in range(1, max_pid + 1):
        source = sources.get(pid)
        proof = SOLVERS_DIR / f"{pid}.py.lean"
        solved_by_pid[pid] = False
        if source is None or not proof.exists():
            continue
        code = lean_code_without_comments(proof.read_text())
        # Some entry modules import their contracts instead of declaring them here.
        solved_by_pid[pid] = bool(code.strip()) and not re.search(
            r"\b(?:axiom|admit|sorry)\b", code
        )
    return solved_by_pid, max_pid


def compute_formal_status_lines() -> list[str]:
    lines: list[str] = []
    for heading, compute in (
        ("Python <<formalsw>>", compute_formalsw_status),
        ("Lean", compute_lean_status),
    ):
        lines.append(f"* {heading}:")
        lines.extend(build_formal_status_lines(*compute()))
    return lines


def update_readme_formal_status(
    lines: list[str], status_lines: list[str]
) -> list[str]:
    for idx, line in enumerate(lines):
        if line.strip() != FORMAL_STATUS_MARKER:
            continue
        start = idx + 1
        end = start
        while end < len(lines) and re.match(r"^\*+ ", lines[end]):
            end += 1
        return lines[:start] + status_lines + lines[end:]
    raise ValueError(f"{FORMAL_STATUS_MARKER} marker not found")


def autoupdate_formal_status(readme_path: Path = README_PATH) -> list[str]:
    lines = readme_path.read_text().splitlines()
    status_lines = compute_formal_status_lines()
    updated_lines = update_readme_formal_status(lines, status_lines)
    readme_path.write_text("\n".join(updated_lines) + "\n")
    return status_lines


if __name__ == "__main__":
    raise SystemExit(main())
