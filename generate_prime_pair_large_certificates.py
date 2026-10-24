#!/usr/bin/env python3
"""
Generate P60's sparse possible-edge graph below the seeded sum budget.

Previously certified small-domain rows are reused. New omitted competitive
pairs require a source rejection; other pairs need only arithmetic bounds.
"""

import argparse
from pathlib import Path

import generate_prime_pair_certificates as base


ROOT = Path(__file__).resolve().parent
DESTINATION = ROOT / "ProjectEulerSolutions/Lib/PrimePairLarge"
BUDGET = 26034
ROWS_PER_FILE = 4


def layout() -> tuple[list[tuple[str, int, int]], list[tuple[str, str, str]], list[tuple[str, int, int]]]:
    boundaries = sorted(set(range(2, base.LARGE_END, base.INTERVAL_SIZE)) | {base.SMALL_END, base.LARGE_END})
    intervals = list(zip(boundaries, boundaries[1:]))
    groups = []
    for selected in ([part for part in intervals if part[1] <= base.SMALL_END], [part for part in intervals if part[0] >= base.SMALL_END]):
        groups.extend(selected[i : i + base.INTERVALS_PER_FILE] for i in range(0, len(selected), base.INTERVALS_PER_FILE))
    blocks = [(f"block_{i:03}", group[0][0], group[-1][1]) for i, group in enumerate(groups)]
    parts = [("small", 2, base.SMALL_END), *(part for part in blocks if part[1] >= base.SMALL_END)]
    merges = []
    assert base.combine([], parts, "large", merges) == ("large", 2, base.LARGE_END)
    return parts, merges, blocks


def choices(p: int, values: list[int]) -> tuple[list[int], list[str]]:
    row = []
    plan = []
    for q in values:
        if q <= p or BUDGET <= p + q:
            plan.append(".outside")
        elif (p + q) % 3 == 0:
            plan.append(".first")
        else:
            first = int(str(p) + str(q))
            second = int(str(q) + str(p))
            first_divisor = next((d for d in (7, 11, 13, 17, 19, 23, 29, 31, 37) if 37 < first and first % d == 0), None)
            second_divisor = next((d for d in (7, 11, 13, 17, 19, 23, 29, 31, 37) if 37 < second and second % d == 0), None)
            if first_divisor is not None:
                plan.append(f"(.firstFactor {first_divisor})")
            elif second_divisor is not None:
                plan.append(f"(.secondFactor {second_divisor})")
            elif not base.source_prime_check(first):
                plan.append(".first")
            elif not base.source_prime_check(second):
                plan.append(".second")
            else:
                row.append(q)
                plan.append(".possible")
    return row, plan


def bounds_file(blocks: list[tuple[str, int, int]]) -> str:
    lines = [
        "import ProjectEulerSolutions.Lib.PrimePairVertices.Trees",
        "import ProjectEulerSolutions.Lib.PrimePairBudgetSearch",
        "namespace ProjectEulerSolutions.PrimePairLarge.Bounds",
    ]
    for name, start, end in blocks:
        lines.extend([
            "set_option maxRecDepth 4096 in",
            f"theorem {name} : ∀ q ∈ PrimePairVertices.tree_{name}.values, {start} ≤ q ∧ q < {end} := by decide",
        ])
    lines.extend([
        "theorem small : ∀ q ∈ PrimePairVertices.tree_small.values, 2 ≤ q ∧ q < 8389 := by",
        "  intro q member",
        "  have shape : PrimePairVertices.tree_small.values =",
        "      (PrimePairVertices.tree_block_000.values ++ PrimePairVertices.tree_block_001.values) ++",
        "        (PrimePairVertices.tree_block_002.values ++ (PrimePairVertices.tree_block_003.values ++ PrimePairVertices.tree_block_004.values)) := rfl",
        "  rw [shape] at member",
        "  simp only [List.mem_append] at member",
        "  rcases member with (first | second) | third | fourth | fifth",
        "  · have := block_000 q first; omega",
        "  · have := block_001 q second; omega",
        "  · have := block_002 q third; omega",
        "  · have := block_003 q fourth; omega",
        "  · have := block_004 q fifth; omega",
        "end ProjectEulerSolutions.PrimePairLarge.Bounds",
    ])
    return "\n".join(lines) + "\n"


def rows_file(group: list[int], domain: list[int], positions: dict[int, int], parts: list[tuple[str, int, int]], merges: list[tuple[str, str, str]]) -> str:
    imports = {"ProjectEulerSolutions.Lib.PrimePairLarge.Bounds"}
    for p in group:
        if p < base.SMALL_END:
            index = positions[p]
            if index < base.SMALL_CACHE_ROWS:
                imports.add("ProjectEulerSolutions.Lib.PrimePairCache.SmallPrefix")
            else:
                imports.add(f"ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows{(index - base.SMALL_CACHE_ROWS) // base.CACHE_ROWS_PER_FILE:03}")
    lines = [*(f"import {name}" for name in sorted(imports)), "", "namespace ProjectEulerSolutions.PrimePairLarge", "open PrimePairCertificate", ""]
    for p in group:
        for name, start, end in parts:
            row = f"row{p}_{name}"
            if name == "small" and p < base.SMALL_END:
                reference = f"PrimePairCache.smallRow{p}" if positions[p] < base.SMALL_CACHE_ROWS else f"PrimePairUpper.smallRow{p}"
                lines.append(f"def {row} : List Nat := {reference}")
                lines.append(f"theorem {row}_covers : Budget.RowUpperBound PrimePairVertices.tree_{name} {BUDGET} {p} {row} :=")
                if positions[p] < base.SMALL_CACHE_ROWS:
                    lines.append(f"  Budget.row_upper_of_exact _ _ _ _ {reference}_checked")
                else:
                    lines.append(f"  Budget.RowUpperBound.of_unbounded {reference}_covers")
            elif end <= p:
                lines.extend([
                    f"def {row} : List Nat := []",
                    f"theorem {row}_covers : Budget.RowUpperBound PrimePairVertices.tree_{name} {BUDGET} {p} {row} :=",
                    f"  Budget.empty_row_before _ _ _ {end} (fun q member => (Bounds.{name} q member).2) (by decide)",
                ])
            elif BUDGET <= p + start:
                lines.extend([
                    f"def {row} : List Nat := []",
                    f"theorem {row}_covers : Budget.RowUpperBound PrimePairVertices.tree_{name} {BUDGET} {p} {row} :=",
                    f"  Budget.empty_row_after _ _ _ {start} (fun q member => (Bounds.{name} q member).1) (by decide)",
                ])
            else:
                values = [q for q in domain if start <= q < end]
                neighbors, plan = choices(p, values)
                lines.extend([
                    f"def {row} : List Nat := {neighbors}",
                    f"def plan{p}_{name} : List Budget.Choice := [{', '.join(plan)}]",
                    "set_option maxRecDepth 2048 in",
                    f"theorem {row}_covers : Budget.RowUpperBound PrimePairVertices.tree_{name} {BUDGET} {p} {row} := by",
                    f"  apply Budget.row_upper_of_plan _ _ _ plan{p}_{name}",
                    "  · decide",
                    "  · rfl",
                ])
            lines.append("")
        for name, left, right in merges:
            row = f"row{p}_{name}"
            lines.extend([
                f"def {row} : List Nat := row{p}_{left} ++ row{p}_{right}",
                f"theorem {row}_covers : Budget.RowUpperBound PrimePairVertices.tree_{name} {BUDGET} {p} {row} :=",
                f"  Budget.RowUpperBound.branch row{p}_{left}_covers row{p}_{right}_covers",
                "",
            ])
        lines.extend([
            f"def row{p} : List Nat := row{p}_large",
            f"theorem row{p}_covers : Budget.RowUpperBound PrimePairVertices.largeTree {BUDGET} {p} row{p} := row{p}_large_covers",
            "",
        ])
    lines.append("end ProjectEulerSolutions.PrimePairLarge")
    return "\n".join(lines) + "\n"


def data_file(keys: list[int], domain: list[int]) -> str:
    lines = [
        "import ProjectEulerSolutions.Lib.PrimePairVertices.Trees",
        "import ProjectEulerSolutions.Lib.PrimePairBudgetSearch",
        "namespace ProjectEulerSolutions.PrimePairLarge.Data",
        "open PrimePairCertificate",
    ]
    for p in keys:
        row = [q for q in domain if p < q and p + q < BUDGET and base.source_pair_check(p, q)]
        lines.append(f"def row{p} : List Nat := {row}")

    def assemble(values: list[int]) -> str:
        if not values:
            return "Graph.empty"
        middle = len(values) // 2
        p = values[middle]
        left = assemble(values[:middle])
        right = assemble(values[middle + 1 :])
        name = f"node{p}"
        lines.append(f"def {name} : Graph := .node {p} row{p} {left} {right}")
        return name

    root = assemble(keys)
    lines.extend([
        f"def graph : Graph := {root}",
        "set_option maxRecDepth 8192 in",
        f"theorem vertices : graph.vertices = PrimePairVertices.largeTree.filter (fun p => decide (2 * p < {BUDGET})) := by decide",
        "theorem coverage (p : Nat) (member : p ∈ PrimePairVertices.large) (competitive : 2 * p < 26034) : p ∈ graph.vertices := by",
        "  rw [vertices, VertexTree.filter_eq, PrimePairVertices.largeTree_values]",
        "  exact List.mem_filter.mpr ⟨member, by simpa only [decide_eq_true_eq] using competitive⟩",
        "end ProjectEulerSolutions.PrimePairLarge.Data",
    ])
    return "\n".join(lines) + "\n"


def graph_file(keys: list[int], module_count: int) -> str:
    lines = [
        "import ProjectEulerSolutions.Lib.PrimePairLarge.Data",
        *(f"import ProjectEulerSolutions.Lib.PrimePairLarge.Rows{i:03}" for i in range(module_count)),
        "namespace ProjectEulerSolutions.PrimePairLarge",
        "open PrimePairCertificate",
        f"theorem empty_covers : Budget.Covers PrimePairVertices.largeTree {BUDGET} Graph.empty := Budget.empty_covers _ _",
    ]

    def assemble(values: list[int]) -> str:
        if not values:
            return "Graph.empty"
        middle = len(values) // 2
        p = values[middle]
        left = assemble(values[:middle])
        right = assemble(values[middle + 1 :])
        left_proof = "empty_covers" if left == "Graph.empty" else f"{left}_covers"
        right_proof = "empty_covers" if right == "Graph.empty" else f"{right}_covers"
        name = f"node{p}"
        lines.extend([
            f"def {name} : Graph := .node {p} row{p} {left} {right}",
            "set_option maxRecDepth 4096 in",
            f"theorem {name}_covers : Budget.Covers PrimePairVertices.largeTree {BUDGET} {name} :=",
            f"  Budget.node_covers _ _ {p} row{p} {left} {right}",
            f"    {left_proof} {right_proof} (by decide) (by decide) row{p}_covers",
        ])
        return name

    graph = assemble(keys)
    lines.extend([
        f"def graph : Graph := {graph}",
        f"theorem graph_covers : Budget.Covers PrimePairVertices.largeTree {BUDGET} graph := {graph}_covers",
        "set_option maxRecDepth 4096 in",
        "theorem graph_eq_data : graph = Data.graph := rfl",
        "theorem graph_coverage (p : Nat) (member : p ∈ PrimePairVertices.large) (competitive : 2 * p < 26034) : p ∈ graph.vertices := by",
        "  rw [graph_eq_data]",
        "  exact Data.coverage p member competitive",
        "end ProjectEulerSolutions.PrimePairLarge",
    ])
    return "\n".join(lines) + "\n"


def search_model(
    adjacency: dict[int, set[int]], need: int, clique: tuple[int, ...], remaining: list[int],
    total: int, best: tuple[int, tuple[int, ...]], stats: dict[str, int] | None = None,
) -> tuple[int, tuple[int, ...]]:
    """
    Produce candidate traces; Lean must still check every resulting claim.
    """
    if stats is not None:
        stats[f"nodes_{need}"] = stats.get(f"nodes_{need}", 0) + 1
        stats[f"max_candidates_{need}"] = max(stats.get(f"max_candidates_{need}", 0), len(remaining))
    if need == 0:
        if total < best[0]:
            if stats is not None:
                stats["improvements"] = stats.get("improvements", 0) + 1
            return total, clique
        return best
    if len(remaining) < need:
        return best
    for index, p in enumerate(remaining):
        if best[0] <= total + p * need:
            break
        candidates = []
        for q in remaining[index + 1 :]:
            if best[0] <= total + p + q:
                break
            if q in adjacency.get(p, set()):
                candidates.append(q)
        if need - 1 <= len(candidates):
            best = search_model(adjacency, need - 1, clique + (p,), candidates, total + p, best, stats)
    return best


def generated_files() -> dict[Path, str]:
    domain = base.prime_values(base.LARGE_END - 1)
    keys = [p for p in domain if 2 * p < BUDGET]
    positions = {p: i for i, p in enumerate(domain)}
    parts, merges, blocks = layout()
    groups = [keys[i : i + ROWS_PER_FILE] for i in range(0, len(keys), ROWS_PER_FILE)]
    files = {DESTINATION / "Bounds.lean": bounds_file(blocks), DESTINATION / "Data.lean": data_file(keys, domain)}
    files.update({DESTINATION / f"Rows{i:03}.lean": rows_file(group, domain, positions, parts, merges) for i, group in enumerate(groups)})
    files[DESTINATION / "Graph.lean"] = graph_file(keys, len(groups))
    return files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare generated files without writing")
    arguments = parser.parse_args()
    files = generated_files()
    stale = []
    for path, content in files.items():
        if arguments.check:
            if not path.exists() or path.read_text() != content:
                stale.append(str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists() or path.read_text() != content:
                path.write_text(content)
    if stale:
        raise SystemExit("Outdated large-domain certificates:\n" + "\n".join(stale))
    print(f"{'Checked' if arguments.check else 'Generated'} {len(files)} large-domain P60 certificate files")
