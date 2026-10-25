#!/usr/bin/env python3
"""
Generate small kernel checks for Euler 62's finite cube-root partition.

Python proposes the rows; Lean checks their signatures, order, and coverage.
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DESTINATION = ROOT / "ProjectEulerSolutions/Lib/CubicPartitionCertificates"
BOUND = 9999
ROWS_PER_LEAF = 128
ROOTS_PER_PART = 64
NAMESPACE = "ProjectEulerSolutions.CubicPartitionCertificates"
IMPORT = "ProjectEulerSolutions.Lib.CubicPartitionCertificates"


@dataclass
class Row:
    key: int
    digits: tuple[int, ...]
    roots: list[int]

    def lean(self) -> str:
        return f"⟨{self.key}, {list(self.digits)}, {self.roots}⟩"


@dataclass
class Branch:
    name: str
    rows: list[Row]
    left: "Branch | None" = None
    right: "Branch | None" = None

    @property
    def tree(self) -> str:
        return f"{self.name}.tree" if self.left is None else self.name

    @property
    def ordered(self) -> str:
        return f"{self.name}.ordered_checked" if self.left is None else f"{self.name}_ordered"

    @property
    def checked(self) -> str:
        return f"{self.name}.rows_checked" if self.left is None else f"{self.name}_rows"

    @property
    def selected(self) -> str:
        return f"{self.name}.selected_checked" if self.left is None else f"{self.name}_selected"

    @property
    def candidates(self) -> list[int]:
        return [row.roots[0] ** 3 for row in self.rows if len(row.roots) == 5]


def partition() -> list[Row]:
    classes: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for root in range(1, BOUND + 1):
        digits = str(root**3)
        classes[tuple(digits.count(str(digit)) for digit in range(10))].append(root)
    rows = [
        Row(sum(count * 14**digit for digit, count in enumerate(digits)), digits, roots)
        for digits, roots in classes.items()
    ]
    rows.sort(key=lambda row: row.key)
    assert len({row.key for row in rows}) == len(rows)
    return rows


def literal_tree(rows: list[Row], depth: int = 0) -> str:
    if not rows:
        return ".empty"
    middle = len(rows) // 2
    row = rows[middle].lean()
    left = literal_tree(rows[:middle], depth + 1)
    right = literal_tree(rows[middle + 1 :], depth + 1)
    indent = "  " * (depth + 1)
    return f"(.node {row}\n{indent}{left}\n{indent}{right})"


def branches(rows: list[Row]) -> tuple[Branch, list[Branch], list[Branch]]:
    leaves: list[Branch] = []
    internal: list[Branch] = []

    def split(values: list[Row]) -> Branch:
        if len(values) <= ROWS_PER_LEAF:
            leaf = Branch(f"Leaf{len(leaves):03}", values)
            leaves.append(leaf)
            return leaf
        middle = len(values) // 2
        left = split(values[:middle])
        right = split(values[middle + 1 :])
        branch = Branch(f"branch{len(internal):03}", values, left, right)
        internal.append(branch)
        return branch

    return split(rows), leaves, internal


def leaf_source(leaf: Branch) -> str:
    return "\n".join(
        [
            "import ProjectEulerSolutions.Lib.CubicPartition",
            "",
            "/-! Proposed classes are checked independently to bound kernel memory. -/",
            f"namespace {NAMESPACE}.{leaf.name}",
            "open ProjectEulerSolutions.CubicPartition",
            f"def tree : Tree := {literal_tree(leaf.rows)}",
            "set_option maxHeartbeats 2000000 in",
            "theorem ordered_checked : ordered tree = true := by decide",
            "set_option maxHeartbeats 2000000 in",
            f"theorem rows_checked : rowsChecked {BOUND} tree = true := by decide",
            "set_option maxHeartbeats 2000000 in",
            f"theorem selected_checked : candidates 5 tree = {leaf.candidates} := by decide",
            f"end {NAMESPACE}.{leaf.name}",
            "",
        ]
    )


def tree_source(root: Branch, leaves: list[Branch], internal: list[Branch]) -> str:
    lines = [*(f"import {IMPORT}.{leaf.name}" for leaf in leaves), "", f"namespace {NAMESPACE}"]
    lines += ["open ProjectEulerSolutions.CubicPartition"]
    for branch in internal:
        left, right = branch.left, branch.right
        assert left is not None and right is not None
        row = branch.rows[len(branch.rows) // 2].lean()
        lines += [
            f"def {branch.name} : Tree := .node {row} {left.tree} {right.tree}",
            "set_option maxHeartbeats 2000000 in",
            f"theorem {branch.ordered} : ordered {branch.tree} = true :=",
            f"  ordered_node _ _ _ {left.ordered} {right.ordered} (by decide)",
            "set_option maxHeartbeats 2000000 in",
            f"theorem {branch.checked} : rowsChecked {BOUND} {branch.tree} = true :=",
            f"  rowsChecked_node _ _ _ _ (by decide) {left.checked} {right.checked}",
            f"theorem {branch.selected} : candidates 5 {branch.tree} = {branch.candidates} := by",
            f"  rw [show {branch.tree} = .node {row} {left.tree} {right.tree} from rfl,",
            f"    candidates_node _ _ _ _ _ _ {left.selected} {right.selected}]",
            "  decide",
        ]
    lines += [
        f"abbrev tree := {root.tree}",
        f"theorem tree_ordered : ordered tree = true := {root.ordered}",
        f"theorem tree_rows : rowsChecked {BOUND} tree = true := {root.checked}",
        f"theorem tree_selected : candidates 5 tree = {root.candidates} := {root.selected}",
        f"end {NAMESPACE}",
        "",
    ]
    return "\n".join(lines)


def part_source(index: int, first: int, count: int) -> str:
    return "\n".join(
        [
            f"import {IMPORT}.Tree",
            "",
            "/-! Every root must be present in its signature's checked class. -/",
            f"namespace {NAMESPACE}.Part{index:03}",
            "open ProjectEulerSolutions.CubicPartition",
            "set_option maxHeartbeats 2000000 in",
            f"theorem checked : Covers tree {first} {count} :=",
            f"  covers_checked tree {first} {count} (by decide)",
            f"end {NAMESPACE}.Part{index:03}",
            "",
        ]
    )


def combine(parts: list[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    middle = len(parts) // 2
    return f"({combine(parts[:middle])}).append ({combine(parts[middle:])})"


def certificate_source(root: Branch, count: int) -> str:
    return "\n".join(
        [
            *(f"import {IMPORT}.Part{index:03}" for index in range(count)),
            "",
            f"namespace {NAMESPACE}",
            "open ProjectEulerSolutions.CubicPartition ProjectEulerSolutions.CubicGroupCounts",
            "open ProjectEulerSolutions.CubicSearch ProjectEulerSolutions.CubicWindows",
            f"theorem covered_all : Covers tree 1 {BOUND} :=",
            f"  {combine([f'Part{index:03}.checked' for index in range(count)])}",
            f"theorem valid : Valid {BOUND} tree :=",
            f"  ⟨tree_ordered, rowsChecked_sound {BOUND} tree tree_rows, covered_all⟩",
            "theorem summary : Summary (prior 10000) ((entries tree).map Row.group) := valid.summary",
            f"theorem matching_checked : ProjectEulerSolutions.CubicGroups.matching 5 ((entries tree).map Row.group) = {root.candidates} :=",
            "  (matching_candidates 5 tree).trans tree_selected",
            "theorem answer_checked : choose 5 ((entries tree).map Row.group) = 127035954683 := by",
            "  rw [choose, matching_checked]",
            "  decide",
            "theorem closed : width 9999 < width 10000 := by decide",
            "theorem search_result : Returns 5 start 127035954683 := by",
            "  have value := summary.choice_unique (collect_summary (prior 10000)) 5",
            "  rw [answer_checked] at value",
            "  have successful : choose 5 (collect (prior 10000)) ≠ 0 := by omega",
            "  have result := returns_closed 5 10000 (by decide) closed successful",
            "  simpa only [← value] using result",
            f"end {NAMESPACE}",
            "",
        ]
    )


def certificate_files() -> dict[Path, str]:
    root, leaves, internal = branches(partition())
    files = {DESTINATION / f"{leaf.name}.lean": leaf_source(leaf) for leaf in leaves}
    files[DESTINATION / "Tree.lean"] = tree_source(root, leaves, internal)
    parts = list(range(1, BOUND + 1, ROOTS_PER_PART))
    for index, first in enumerate(parts):
        files[DESTINATION / f"Part{index:03}.lean"] = part_source(index, first, min(ROOTS_PER_PART, BOUND + 1 - first))
    files[DESTINATION / "Certificate.lean"] = certificate_source(root, len(parts))
    return files


def generate(check: bool) -> None:
    files = certificate_files()
    stale = [path for path, contents in files.items() if not path.exists() or path.read_text() != contents]
    extra = set(DESTINATION.glob("*.lean")) - files.keys()
    if check:
        if stale or extra:
            raise SystemExit("Cubic permutation certificates are out of date")
    else:
        DESTINATION.mkdir(parents=True, exist_ok=True)
        for path in stale:
            path.write_text(files[path])
        for path in extra:
            path.unlink()
    print(f"{'Checked' if check else 'Generated'} {len(files)} cubic permutation certificate files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare generated sources without writing")
    arguments = parser.parse_args()
    generate(arguments.check)
