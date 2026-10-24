#!/usr/bin/env python3
"""
Generate the bounded P60 search trace for the smaller-domain graph.

Every candidate list and child result still requires a Lean kernel check.
Numeric modules stay independent of the preceding outer scan certificates.
"""

import argparse
from pathlib import Path

import generate_prime_pair_certificates as base


ROOT = Path(__file__).resolve().parent
DESTINATION = ROOT / "ProjectEulerSolutions/Lib/PrimePairUpper"
ROOTS_PER_FILE = 8


def search_files(domain: list[int]) -> dict[Path, str]:
    neighbors = {p: [q for q in domain if p < q and base.source_pair_check(p, q)] for p in domain}
    sets = {p: set(row) for p, row in neighbors.items()}
    positions = {p: i for i, p in enumerate(domain)}

    def row_reference(p: int) -> str:
        return f"PrimePairCache.smallRow{p}" if positions[p] < base.SMALL_CACHE_ROWS else f"SearchData.row{p}"

    setup = [
        "import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchData",
        "import ProjectEulerSolutions.Lib.PrimePairLookup",
        "",
        "namespace ProjectEulerSolutions.PrimePairUpper.Search",
        "open CliqueSearch",
        "def edge : Nat → Nat → Bool := SearchData.graph.lookupEdge",
        "def best : Best := ⟨10 ^ 30, []⟩",
        f"def tail{len(domain)} : List Nat := []",
        *(f"def tail{i} : List Nat := {domain[i]} :: tail{i + 1}" for i in reversed(range(len(domain)))),
        "set_option maxRecDepth 4096 in",
        "theorem tail0_eq_vertices : tail0 = PrimePairVertices.small := by decide",
        f"theorem outerScan{len(domain)} : scan edge (search edge 4) 5 [] 0 tail{len(domain)} best = best := rfl",
        "end ProjectEulerSolutions.PrimePairUpper.Search",
    ]
    output = {DESTINATION / "SmallSearchSetup.lean": "\n".join(setup) + "\n"}
    groups = []
    stop = len(domain)
    while stop:
        start = stop
        steps = 0
        while start and stop - start < ROOTS_PER_FILE:
            cost = max(1, len(neighbors[domain[start - 1]]))
            if start < stop and steps + cost > 64:
                break
            start -= 1
            steps += cost
        groups.append((start, stop))
        stop = start
    for chunk, (start, stop) in enumerate(groups):
        lines = [
            "import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup",
            "",
            "/-! Split bounded search: every child is checked separately. -/",
            "namespace ProjectEulerSolutions.PrimePairUpper.Search",
            "open CliqueSearch",
            "",
        ]
        scans = [
            f"import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch{chunk - 1:03}"
            if chunk else "import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup",
            f"import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch{chunk:03}",
            "",
            "namespace ProjectEulerSolutions.PrimePairUpper.Search",
            "open CliqueSearch",
            "",
        ]
        for i in reversed(range(start, stop)):
            p = domain[i]
            row = neighbors[p]
            size = len(row)
            lines.extend([
                f"def rootCandidates{i} : List Nat := {row}",
                "set_option maxRecDepth 4096 in",
                f"theorem rootCandidates{i}_checked : candidates edge 5 0 {p} tail{i + 1} best.bound = rootCandidates{i} := by",
                "  unfold candidates",
                f"  have row : edge {p} = {row_reference(p)}.contains := by funext q; rfl",
                "  rw [row]",
                "  decide",
                "",
            ])
            if size >= 4:
                lines.extend([
                    f"def innerTail{i}_{size} : List Nat := []",
                    f"theorem innerScan{i}_{size} : scan edge (search edge 3) 4 [{p}] {p} innerTail{i}_{size} best = best := rfl",
                    "",
                ])
                for j in reversed(range(size)):
                    q = row[j]
                    common = [r for r in row[j + 1 :] if r in sets[q]]
                    lines.extend([
                        f"def innerTail{i}_{j} : List Nat := {q} :: innerTail{i}_{j + 1}",
                        f"def innerCandidates{i}_{j} : List Nat := {common}",
                        "set_option maxRecDepth 4096 in",
                        f"theorem innerCandidates{i}_{j}_checked : candidates edge 4 {p} {q} innerTail{i}_{j + 1} best.bound = innerCandidates{i}_{j} := by",
                        "  unfold candidates",
                        f"  have row : edge {q} = {row_reference(q)}.contains := by funext r; rfl",
                        "  rw [row]",
                        "  decide",
                        "",
                    ])
                    if len(common) >= 3:
                        lines.extend([
                            "set_option maxRecDepth 4096 in",
                            f"theorem innerChild{i}_{j} : search edge 3 [{p}, {q}] innerCandidates{i}_{j} {p + q} best = best := by decide",
                            "",
                        ])
                    lines.extend([
                        f"theorem innerScan{i}_{j} : scan edge (search edge 3) 4 [{p}] {p} innerTail{i}_{j} best = best := by",
                        f"  rw [innerTail{i}_{j}, scan, if_neg (by decide), innerCandidates{i}_{j}_checked]",
                        f"  change scan edge (search edge 3) 4 [{p}] {p} innerTail{i}_{j + 1}",
                        f"    (if 3 ≤ innerCandidates{i}_{j}.length then search edge 3 [{p}, {q}] innerCandidates{i}_{j} {p + q} best else best) = best",
                        f"  rw [if_pos (by decide), innerChild{i}_{j}]" if len(common) >= 3 else "  rw [if_neg (by decide)]",
                        f"  exact innerScan{i}_{j + 1}",
                        "",
                    ])
                lines.extend([
                    f"theorem rootBranch{i} : search edge 4 [{p}] rootCandidates{i} {p} best = best := by",
                    "  rw [search_succ_eq_scan]",
                    f"  change scan edge (search edge 3) 4 [{p}] {p} innerTail{i}_0 best = best",
                    f"  exact innerScan{i}_0",
                    "",
                ])
            scans.extend([
                "set_option maxRecDepth 4096 in",
                f"theorem outerScan{i} : scan edge (search edge 4) 5 [] 0 tail{i} best = best := by",
                f"  have cons : tail{i} = {p} :: tail{i + 1} := rfl",
                f"  rw [cons, scan, if_neg (by decide), rootCandidates{i}_checked]",
                f"  change scan edge (search edge 4) 5 [] 0 tail{i + 1}",
                f"    (if 4 ≤ rootCandidates{i}.length then search edge 4 [{p}] rootCandidates{i} {p} best else best) = best",
                f"  rw [if_pos (by decide), rootBranch{i}]" if size >= 4 else "  rw [if_neg (by decide)]",
                f"  exact outerScan{i + 1}",
                "",
            ])
        if start == 0:
            scans.extend([
                "theorem smallSearch : search edge 5 [] PrimePairVertices.small 0 best = best := by",
                "  rw [search_succ_eq_scan, ← tail0_eq_vertices]",
                "  exact outerScan0",
                "theorem smallSearch_edge : search SearchData.graph.edge 5 [] PrimePairVertices.small 0 best = best := by",
                "  rw [PrimePairCertificate.Graph.edge_eq_lookup]",
                "  exact smallSearch",
                "",
            ])
        lines.append("end ProjectEulerSolutions.PrimePairUpper.Search")
        scans.append("end ProjectEulerSolutions.PrimePairUpper.Search")
        output[DESTINATION / f"SmallBranch{chunk:03}.lean"] = "\n".join(lines) + "\n"
        output[DESTINATION / f"SmallSearch{chunk:03}.lean"] = "\n".join(scans) + "\n"
    output[DESTINATION / "SmallCertificate.lean"] = "\n".join([
        "import ProjectEulerSolutions.Lib.PrimePairUpper.SmallGraph",
        f"import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch{chunk:03}",
        "",
        "namespace ProjectEulerSolutions.PrimePairUpper",
        "theorem small_search_checked : CliqueSearch.search smallGraph.edge 5 [] PrimePairVertices.small 0",
        "    ⟨10 ^ 30, []⟩ = ⟨10 ^ 30, []⟩ := by",
        "  rw [smallGraph_eq_searchData]",
        "  exact Search.smallSearch_edge",
        "end ProjectEulerSolutions.PrimePairUpper",
        "",
    ])
    return output


def generated_files() -> dict[Path, str]:
    return search_files(base.prime_values(base.SMALL_END - 1))


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
        raise SystemExit("Outdated smaller-domain search certificates:\n" + "\n".join(stale))
    print(f"{'Checked' if arguments.check else 'Generated'} {len(files)} P60 smaller-domain search certificate files")
