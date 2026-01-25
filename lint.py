#!/usr/bin/env python3
from __future__ import annotations

import io
import argparse
import re
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOLUTIONS_PATH = ROOT / "data/projecteuler-solutions/Solutions.md"
SOLVERS_DIR = ROOT / "solvers"

LINE_RE = re.compile(r"^(\d+)\.\s+(.*)$")
FORBIDDEN_TOKENS = ("Solutions.md",)


@dataclass(frozen=True)
class Violation:
    kind: str
    puzzle_id: int
    path: Path
    answer: str
    context: list[tuple[int, str]]


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
    if "_" in stem:
        prefix = stem.split("_", 1)[0]
        if prefix.isdigit():
            return int(prefix)
    return None


def iter_solver_sources() -> list[Path]:
    sources: list[Path] = []
    for ext in (".py", ".c", ".cpp"):
        sources.extend(SOLVERS_DIR.glob(f"*{ext}"))
    return sorted(sources)


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
        if answer not in line:
            continue
        if re.search(r"\breturn\b", line):
            hits.append(idx)
            continue
        if re.search(r"\bprintf\s*\(", line):
            hits.append(idx)
            continue
    return hits


def forbidden_hits(text: str, token: str) -> list[int]:
    return [idx for idx, line in enumerate(text.splitlines(), 1) if token in line]


def lint_paths(
    paths: list[Path], answers: dict[int, str] | None = None
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
            normalized_text = " ".join(text.split())
            theorem_re = re.compile(
                rf"theorem equiv\b .*? : "
                rf"ProjectEulerStatements\.P{pid}\.naive (.+?) = solve \1 := "
            )
            if not theorem_re.search(normalized_text):
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
            last_line = lines[-1] if lines else ""
            if not re.match(
                r"^\s*IO\.println \((?:solve|serialize \(solve) [^\s\)]", last_line
            ):
                context: list[tuple[int, str]] = []
                if lines:
                    context = [(len(lines), last_line.rstrip())]
                violations.append(
                    Violation(
                        "lean",
                        pid,
                        path,
                        "last line must start with 'IO.println (solve ' and include an argument",
                        context,
                    )
                )
            continue
        answer = answers.get(pid)
        if not answer or not should_scan_answer(answer):
            continue
        if path.suffix not in (".py", ".c", ".cpp"):
            continue
        if path.suffix != ".py":
            try:
                text = path.read_text()
            except OSError as exc:
                print(f"error: failed to read {path}: {exc}", file=sys.stderr)
                continue
        if answer and should_scan_answer(answer):
            if path.suffix == ".py":
                line_hits = python_comment_or_string_hits(text, answer)
            else:
                line_hits = c_comment_hits(text, answer)
                line_hits += c_line_hits(text, answer)
            if line_hits:
                lines = text.splitlines()
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
            lines = text.splitlines()
            hit_lines = sorted(set(token_hits))
            context = [
                (line_no, lines[line_no - 1].rstrip())
                for line_no in hit_lines
                if 0 < line_no <= len(lines)
            ]
            violations.append(Violation("forbidden", pid, path, token, context))
    return violations


def format_violations(violations: list[Violation], root: Path = ROOT) -> list[str]:
    if not violations:
        return ["ok: no forbidden content found in solver sources."]
    lines = ["error: forbidden content found in solver sources:"]
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
        elif violation.kind == "lean":
            detail = violation.answer
        else:
            detail = f"contains reference answer {violation.answer!r}"
        lines.append(f"- {violation.puzzle_id}: {rel} {detail}")
        for line_no, line in violation.context:
            lines.append(f"  line {line_no}: {line}")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check solver sources for forbidden content."
    )
    parser.add_argument(
        "-l",
        "--language",
        choices=("py", "c", "cpp", "lean"),
        help="Only lint a specific language.",
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
        paths = iter_solver_sources()
    if args.language == "lean":
        if args.paths:
            paths = [path for path in paths if path.suffix == ".lean"]
        else:
            paths = sorted(SOLVERS_DIR.glob("*.lean"))
    elif args.language:
        lang_ext = f".{args.language}"
        paths = [path for path in paths if path.suffix == lang_ext]
    else:
        paths = [path for path in paths if path.suffix != ".lean"]
    violations = lint_paths(paths)
    output_lines = format_violations(violations)
    if violations:
        for line in output_lines:
            print(line)
        return 1
    print(output_lines[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
