#!/usr/bin/env python3
"""
Generate possible-edge rows for P60's remaining smaller-domain vertices.

Lean checks a rejection for every omitted increasing edge. Included edges
remain possible until the bounded search and winning witness are checked.
"""

import argparse
from pathlib import Path

import generate_prime_pair_certificates as base


ROOT = Path(__file__).resolve().parent
DESTINATION = ROOT / "ProjectEulerSolutions/Lib/PrimePairUpper"
ROWS_PER_FILE = 8


def small_blocks() -> tuple[list[tuple[str, int, int]], list[tuple[str, str, str]]]:
    boundaries = sorted(set(range(2, base.LARGE_END, base.INTERVAL_SIZE)) | {base.SMALL_END, base.LARGE_END})
    intervals = [(start, end) for start, end in zip(boundaries, boundaries[1:]) if end <= base.SMALL_END]
    groups = [intervals[i : i + base.INTERVALS_PER_FILE] for i in range(0, len(intervals), base.INTERVALS_PER_FILE)]
    blocks = [(f"block_{i:03}", group[0][0], group[-1][1]) for i, group in enumerate(groups)]
    merges = []
    assert base.combine([], blocks, "small", merges) == ("small", 2, base.SMALL_END)
    return blocks, merges


def row_choices(p: int, values: list[int]) -> tuple[list[int], list[str]]:
    neighbors = []
    choices = []
    for q in values:
        if q <= p or (p + q) % 3 == 0:
            choices.append(".first")
        elif base.source_pair_check(p, q):
            neighbors.append(q)
            choices.append(".possible")
        elif base.source_prime_check(int(str(p) + str(q))):
            choices.append(".second")
        else:
            choices.append(".first")
    return neighbors, choices


def rows_file(group: list[int], domain: list[int], blocks: list[tuple[str, int, int]], merges: list[tuple[str, str, str]]) -> str:
    lines = [
        "import ProjectEulerSolutions.Lib.PrimePairVertices.Trees",
        "import ProjectEulerSolutions.Lib.PrimePairUpperBound",
        "",
        "/-! Generated possible-edge rows; every excluded edge is kernel checked. -/",
        "namespace ProjectEulerSolutions.PrimePairUpper",
        "open PrimePairCertificate",
        "",
    ]
    for p in group:
        for name, start, end in blocks:
            values = [q for q in domain if start <= q < end]
            neighbors, choices = row_choices(p, values)
            row = f"smallRow{p}_{name}"
            plan = f"smallPlan{p}_{name}"
            lines.extend([
                f"def {row} : List Nat := {neighbors}",
                f"def {plan} : List RowChoice := [{', '.join(choices)}]",
                "set_option maxRecDepth 2048 in",
                f"theorem {row}_covers : RowUpperBound PrimePairVertices.tree_{name} {p} {row} := by",
                f"  apply row_upper_of_plan _ _ {plan}",
                "  · decide",
                "  · rfl",
                "",
            ])
        for name, left, right in merges:
            row = f"smallRow{p}_{name}"
            left_row = f"smallRow{p}_{left}"
            right_row = f"smallRow{p}_{right}"
            lines.extend([
                f"def {row} : List Nat := {left_row} ++ {right_row}",
                f"theorem {row}_covers : RowUpperBound PrimePairVertices.tree_{name} {p} {row} :=",
                f"  RowUpperBound.branch {left_row}_covers {right_row}_covers",
                "",
            ])
        lines.extend([
            f"def smallRow{p} : List Nat := smallRow{p}_small",
            f"theorem smallRow{p}_covers : RowUpperBound PrimePairVertices.smallTree {p} smallRow{p} := smallRow{p}_small_covers",
            "",
        ])
    lines.append("end ProjectEulerSolutions.PrimePairUpper")
    return "\n".join(lines) + "\n"


def search_data_file(selected: list[int], domain: list[int]) -> str:
    lines = [
        "import ProjectEulerSolutions.Lib.PrimePairCache.SmallPrefix",
        "",
        "/-! Data-only search view; SmallGraph supplies its source-edge coverage proof. -/",
        "namespace ProjectEulerSolutions.PrimePairUpper.SearchData",
        "open PrimePairCertificate",
        "",
    ]
    for p in selected:
        neighbors = [q for q in domain if p < q and base.source_pair_check(p, q)]
        lines.append(f"def row{p} : List Nat := {neighbors}")
    lines.append("")

    def assemble(values: list[int], with_cache: bool) -> str:
        if not values:
            return "PrimePairCache.smallPrefix" if with_cache else "Graph.empty"
        middle = len(values) // 2
        p = values[middle]
        left = assemble(values[:middle], with_cache)
        right = assemble(values[middle + 1 :], False)
        name = f"node{p}"
        lines.append(f"def {name} : Graph := .node {p} row{p} {left} {right}")
        return name

    root = assemble(selected, True)
    lines.extend([
        f"def graph : Graph := {root}",
        "set_option maxRecDepth 4096 in",
        "theorem vertices : graph.vertices = PrimePairVertices.small := by decide",
        "",
        "end ProjectEulerSolutions.PrimePairUpper.SearchData",
    ])
    return "\n".join(lines) + "\n"


def graph_file(selected: list[int], module_count: int) -> str:
    lines = [
        "import ProjectEulerSolutions.Lib.PrimePairCache.SmallPrefix",
        "import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchData",
        *(f"import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows{i:03}" for i in range(module_count)),
        "",
        "namespace ProjectEulerSolutions.PrimePairUpper",
        "open PrimePairCertificate",
        "",
        "def smallCached : Graph := PrimePairCache.smallPrefix",
        "theorem smallCached_covers : Graph.Covers PrimePairVertices.smallTree smallCached :=",
        "  Graph.Agrees.covers PrimePairCache.smallPrefix_agrees",
        "def smallEmpty : Graph := .empty",
        "theorem smallEmpty_covers : Graph.Covers PrimePairVertices.smallTree smallEmpty := Graph.empty_covers _",
        "",
    ]

    def assemble(values: list[int], with_cache: bool) -> str:
        if not values:
            return "smallCached" if with_cache else "smallEmpty"
        middle = len(values) // 2
        p = values[middle]
        left = assemble(values[:middle], with_cache)
        right = assemble(values[middle + 1 :], False)
        name = f"smallNode{p}"
        lines.extend([
            f"def {name} : Graph := .node {p} smallRow{p} {left} {right}",
            "set_option maxRecDepth 4096 in",
            f"theorem {name}_covers : Graph.Covers PrimePairVertices.smallTree {name} :=",
            f"  Graph.node_covers _ {p} smallRow{p} {left} {right}",
            f"    {left}_covers {right}_covers (by decide) (by decide) smallRow{p}_covers",
            "",
        ])
        return name

    graph = assemble(selected, True)
    lines.extend([
        f"def smallGraph : Graph := {graph}",
        f"theorem smallGraph_covers : Graph.Covers PrimePairVertices.smallTree smallGraph := {graph}_covers",
        "set_option maxRecDepth 4096 in",
        "theorem smallGraph_eq_searchData : smallGraph = SearchData.graph := rfl",
        "set_option maxRecDepth 4096 in",
        "theorem smallGraph_vertices : smallGraph.vertices = PrimePairVertices.small := by",
        "  rw [smallGraph_eq_searchData]",
        "  exact SearchData.vertices",
        "theorem smallGraph_coverage (p : Nat) (member : p ∈ PrimePairVertices.small) : p ∈ smallGraph.vertices := by",
        "  simpa only [smallGraph_vertices] using member",
        "",
        "end ProjectEulerSolutions.PrimePairUpper",
    ])
    return "\n".join(lines) + "\n"


def generated_files() -> dict[Path, str]:
    domain = base.prime_values(base.SMALL_END - 1)
    selected = domain[base.SMALL_CACHE_ROWS :]
    blocks, merges = small_blocks()
    groups = [selected[i : i + ROWS_PER_FILE] for i in range(0, len(selected), ROWS_PER_FILE)]
    files = {DESTINATION / f"SmallRows{i:03}.lean": rows_file(group, domain, blocks, merges) for i, group in enumerate(groups)}
    files[DESTINATION / "SmallGraph.lean"] = graph_file(selected, len(groups))
    files[DESTINATION / "SmallSearchData.lean"] = search_data_file(selected, domain)
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
        raise SystemExit("Outdated possible-edge certificates:\n" + "\n".join(stale))
    print(f"{'Checked' if arguments.check else 'Generated'} {len(files)} P60 possible-edge certificate files")
