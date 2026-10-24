#!/usr/bin/env python3
"""
Generate the seeded P60 search trace for the sparse large-domain graph.

Every candidate list, recursive result, and incumbent change is checked by
Lean. Numeric chunks import no preceding search certificates, limiting memory.
"""

import argparse
from pathlib import Path

import generate_prime_pair_certificates as base
import generate_prime_pair_large_certificates as large


ROOT = Path(__file__).resolve().parent
DESTINATION = ROOT / "ProjectEulerSolutions/Lib/PrimePairLarge"
STEPS_PER_FILE = 32
ROOTS_PER_SCAN_FILE = 8
DIRECT_CANDIDATES = 16
SEED = (26034, ())
WINNER = (26033, (13, 5197, 5701, 6733, 8389))
HEADER = [
    "namespace ProjectEulerSolutions.PrimePairLarge.Search",
    "open CliqueSearch",
]
FOOTER = "end ProjectEulerSolutions.PrimePairLarge.Search"


def best_term(best: tuple[int, tuple[int, ...]]) -> str:
    assert best in (SEED, WINNER), f"Unexpected incumbent: {best}"
    return "seed" if best == SEED else "winner"


class Trace:
    def __init__(self, domain: list[int]):
        self.domain = domain
        self.adjacency = {
            p: {q for q in domain if p < q and p + q < large.BUDGET and base.source_pair_check(p, q)}
            for p in domain if 2 * p < large.BUDGET
        }

    def candidates(self, total: int, p: int, remaining: list[int], best: tuple[int, tuple[int, ...]]) -> list[int]:
        return [q for q in remaining if total + p + q < best[0] and q in self.adjacency.get(p, set())]

    def fast_row_files(self) -> dict[Path, str]:
        def tree(values: list[int]) -> str:
            if not values:
                return ".empty"
            middle = len(values) // 2
            return f"(.node {values[middle]} {tree(values[:middle])} {tree(values[middle + 1 :])})"

        keys = list(self.adjacency)
        files = {}
        modules = []
        for chunk, start in enumerate(range(0, len(keys), 64)):
            module = f"FastRows{chunk:03}"
            modules.append(module)
            lines = [
                "import ProjectEulerSolutions.Lib.PrimePairLarge.Data",
                "import ProjectEulerSolutions.Lib.SortedNatTree",
                "namespace ProjectEulerSolutions.PrimePairLarge.FastRows",
            ]
            for p in keys[start : start + 64]:
                lines.extend([
                    f"def row{p} : SortedNatTree.Tree := {tree(sorted(self.adjacency[p]))}",
                    "set_option maxRecDepth 4096 in",
                    f"theorem row{p}_ordered : SortedNatTree.ordered row{p} = true := by decide",
                    "set_option maxRecDepth 4096 in",
                    f"theorem row{p}_values : SortedNatTree.values row{p} = Data.row{p} := rfl",
                    f"theorem row{p}_checked : SortedNatTree.contains row{p} = Data.row{p}.contains := by",
                    f"  rw [SortedNatTree.contains_eq_list row{p} row{p}_ordered, row{p}_values]",
                ])
            lines.append("end ProjectEulerSolutions.PrimePairLarge.FastRows")
            files[DESTINATION / f"{module}.lean"] = "\n".join([*lines, ""])
        files[DESTINATION / "FastRows.lean"] = "\n".join([
            *(f"import ProjectEulerSolutions.Lib.PrimePairLarge.{module}" for module in modules), "",
        ])
        return files

    def candidate_proof(self, name: str, need: int, total: int, p: int, tail: str, values: list[int], best: tuple[int, tuple[int, ...]], remaining: list[int]) -> list[str]:
        assert p in self.adjacency, f"No stored competitive row for {p}"
        predicate = f"(SortedNatTree.contains FastRows.row{p})"
        row_proof = [
            f"  have row : edge {p} = {predicate} := by",
            f"    change Data.row{p}.contains = {predicate}",
            f"    exact FastRows.row{p}_checked.symm",
        ]
        if len(remaining) > 256:
            prefix_length = sum(total + p + q < best[0] for q in remaining)
            tail_prefix = tail.rstrip("0123456789")
            tail_index = int(tail[len(tail_prefix) :])
            lines = [f"def {name} : List Nat := {values}"]
            steps = []
            for chunk, start in enumerate(range(0, prefix_length, 128)):
                chunk_name = f"{name}Block{chunk}"
                end = min(start + 128, prefix_length)
                first_tail = f"{tail_prefix}{tail_index + start}"
                next_tail = f"{tail_prefix}{tail_index + end}"
                items = remaining[start:end]
                neighbors = [q for q in items if q in self.adjacency[p]]
                lines.extend([
                    f"def {chunk_name} : List Nat := {items}",
                    "set_option maxRecDepth 4096 in",
                    f"theorem {chunk_name}_shape : {first_tail} = {chunk_name} ++ {next_tail} := rfl",
                    "set_option maxRecDepth 4096 in",
                    f"theorem {chunk_name}_checked : {chunk_name}.filter {predicate} = {neighbors} := by decide",
                    "set_option maxRecDepth 4096 in",
                    f"theorem {chunk_name}_step : candidates edge {need} {total} {p} {first_tail} {best_term(best)}.bound =",
                    f"    {neighbors} ++ candidates edge {need} {total} {p} {next_tail} {best_term(best)}.bound := by",
                    f"  rw [{chunk_name}_shape, candidates_append_of_within edge {need} {total} {p} {chunk_name} {next_tail} {best_term(best)}.bound (by decide)]",
                    *row_proof,
                    f"  rw [row, {chunk_name}_checked]",
                ])
                steps.append(f"{chunk_name}_step")
            final_tail = f"{tail_prefix}{tail_index + prefix_length}"
            lines.extend([
                f"theorem {name}_end : candidates edge {need} {total} {p} {final_tail} {best_term(best)}.bound = [] := by decide",
                "set_option maxRecDepth 4096 in",
                f"theorem {name}_checked : candidates edge {need} {total} {p} {tail} {best_term(best)}.bound = {name} := by",
                f"  rw [{', '.join([*steps, name + '_end'])}]",
                "  rfl",
            ])
            return lines
        return [
            f"def {name} : List Nat := {values}",
            "set_option maxRecDepth 8192 in",
            f"theorem {name}_checked : candidates edge {need} {total} {p} {tail} {best_term(best)}.bound = {name} := by",
            "  unfold candidates",
            *row_proof,
            "  rw [row]",
            "  decide",
        ]

    def emit_child(self, lines: list[str], name: str, value_name: str, need: int, clique: tuple[int, ...], remaining: list[int], total: int, before: tuple[int, tuple[int, ...]]) -> tuple[int, tuple[int, ...]]:
        after = large.search_model(self.adjacency, need, clique, remaining, total, before)
        statement = f"search edge {need} {list(clique)} {value_name} {total} {best_term(before)} = {best_term(after)}"
        if len(remaining) <= DIRECT_CANDIDATES:
            lines.extend([
                "set_option maxRecDepth 4096 in",
                f"theorem {name} : {statement} := by decide",
            ])
            return after

        # The largest three-pick intersections exceed the direct-check budget.
        # Split them too, rather than increasing the compiler's memory limit.
        assert need > 0
        lines.append(f"def {name}Tail{len(remaining)} : List Nat := []")
        lines.extend(f"def {name}Tail{i} : List Nat := {remaining[i]} :: {name}Tail{i + 1}" for i in reversed(range(len(remaining))))
        steps = []
        current = before
        for i, p in enumerate(remaining):
            if current[0] <= total + p * need:
                break
            next_values = self.candidates(total, p, remaining[i + 1 :], current)
            candidate = f"{name}Candidates{i}"
            lines.extend(self.candidate_proof(candidate, need, total, p, f"{name}Tail{i + 1}", next_values, current, remaining[i + 1 :]))
            updated = current
            if need - 1 <= len(next_values):
                updated = self.emit_child(lines, f"{name}Child{i}", candidate, need - 1, clique + (p,), next_values, total + p, current)
            steps.append((i, p, candidate, current, updated, need - 1 <= len(next_values)))
            current = updated
        stop = len(steps)
        assert current == after
        terminal = "rfl" if stop == len(remaining) else f"by rw [{name}Tail{stop}, scan, if_pos (by decide)]"
        lines.append(f"theorem {name}Scan{stop} : scan edge (search edge {need - 1}) {need} {list(clique)} {total} {name}Tail{stop} {best_term(after)} = {best_term(after)} := {terminal}")
        for i, p, candidate, prior, updated, enough in reversed(steps):
            lines.extend([
                f"theorem {name}Scan{i} : scan edge (search edge {need - 1}) {need} {list(clique)} {total} {name}Tail{i} {best_term(prior)} = {best_term(after)} := by",
                f"  rw [{name}Tail{i}, scan, if_neg (by decide), {candidate}_checked]",
                f"  change scan edge (search edge {need - 1}) {need} {list(clique)} {total} {name}Tail{i + 1}",
                f"    (if {need - 1} ≤ {candidate}.length then search edge {need - 1} {list(clique + (p,))} {candidate} {total + p} {best_term(prior)} else {best_term(prior)}) = {best_term(after)}",
                f"  rw [if_pos (by decide), {name}Child{i}]" if enough else "  rw [if_neg (by decide)]",
                f"  exact {name}Scan{i + 1}",
            ])
        lines.extend([
            f"theorem {name} : {statement} := by",
            "  rw [search_succ_eq_scan]",
            f"  change scan edge (search edge {need - 1}) {need} {list(clique)} {total} {name}Tail0 {best_term(before)} = {best_term(after)}",
            f"  exact {name}Scan0",
        ])
        return after

    def root_files(self, index: int, before: tuple[int, tuple[int, ...]]) -> tuple[dict[Path, str], tuple[int, tuple[int, ...]]]:
        p = self.domain[index]
        values = self.candidates(0, p, self.domain[index + 1 :], before)
        after = large.search_model(self.adjacency, 4, (p,), values, p, before)
        prefix = f"root{index}"
        setup_module = f"SearchRoot{index:03}Data"
        setup = ["import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup", *HEADER]
        setup.extend([
            f"def {prefix}Candidates : List Nat := Data.row{p}",
            "set_option maxRecDepth 8192 in",
            f"theorem {prefix}Candidates_checked : candidates edge 5 0 {p} tail{index + 1} {best_term(before)}.bound = {prefix}Candidates := by",
            f"  apply candidates_eq_row edge 5 0 {p} tail{index + 1} {prefix}Candidates {best_term(before)}.bound tail{index + 1}_ordered",
            "  · decide",
            "  · decide",
            "  · funext q",
            "    rfl",
        ])
        setup.append(f"def {prefix}Tail{len(values)} : List Nat := []")
        setup.extend(f"def {prefix}Tail{j} : List Nat := {values[j]} :: {prefix}Tail{j + 1}" for j in reversed(range(len(values))))
        files = {DESTINATION / f"{setup_module}.lean": "\n".join([*setup, FOOTER, ""])}
        numeric = []
        steps = []
        current = before
        for j, q in enumerate(values):
            if current[0] <= p + q * 4:
                break
            candidate = f"{prefix}Candidates{j}"
            common = self.candidates(p, q, values[j + 1 :], current)
            declarations = self.candidate_proof(candidate, 4, p, q, f"{prefix}Tail{j + 1}", common, current, values[j + 1 :])
            updated = current
            enough = 3 <= len(common)
            if enough:
                updated = self.emit_child(declarations, f"{prefix}Child{j}", candidate, 3, (p, q), common, p + q, current)
            declarations.extend([
                f"theorem {prefix}Step{j} : scan edge (search edge 3) 4 [{p}] {p} {prefix}Tail{j} {best_term(current)} =",
                f"    scan edge (search edge 3) 4 [{p}] {p} {prefix}Tail{j + 1} {best_term(updated)} := by",
                f"  rw [{prefix}Tail{j}, scan, if_neg (by decide), {candidate}_checked]",
                f"  change scan edge (search edge 3) 4 [{p}] {p} {prefix}Tail{j + 1}",
                f"    (if 3 ≤ {candidate}.length then search edge 3 [{p}, {q}] {candidate} {p + q} {best_term(current)} else {best_term(current)}) = _",
                f"  rw [if_pos (by decide), {prefix}Child{j}]" if enough else "  rw [if_neg (by decide)]",
            ])
            numeric.append(declarations)
            steps.append((j, current))
            current = updated
        assert current == after
        numeric_modules = []
        for chunk, start in enumerate(range(0, len(numeric), STEPS_PER_FILE)):
            module = f"SearchRoot{index:03}Part{chunk:02}"
            numeric_modules.append(module)
            lines = [f"import ProjectEulerSolutions.Lib.PrimePairLarge.{setup_module}", *HEADER]
            for declarations in numeric[start : start + STEPS_PER_FILE]:
                lines.extend(declarations)
            files[DESTINATION / f"{module}.lean"] = "\n".join([*lines, FOOTER, ""])
        assembly = [
            f"import ProjectEulerSolutions.Lib.PrimePairLarge.{setup_module}",
            *(f"import ProjectEulerSolutions.Lib.PrimePairLarge.{module}" for module in numeric_modules),
            *HEADER,
        ]
        stop = len(steps)
        terminal = "rfl" if stop == len(values) else f"by rw [{prefix}Tail{stop}, scan, if_pos (by decide)]"
        assembly.append(f"theorem {prefix}Scan{stop} : scan edge (search edge 3) 4 [{p}] {p} {prefix}Tail{stop} {best_term(after)} = {best_term(after)} := {terminal}")
        for j, prior in reversed(steps):
            assembly.extend([
                f"theorem {prefix}Scan{j} : scan edge (search edge 3) 4 [{p}] {p} {prefix}Tail{j} {best_term(prior)} = {best_term(after)} :=",
                f"  ({prefix}Step{j}).trans {prefix}Scan{j + 1}",
            ])
        assembly.extend([
            "set_option maxRecDepth 4096 in",
            f"theorem {prefix}Branch : search edge 4 [{p}] {prefix}Candidates {p} {best_term(before)} = {best_term(after)} := by",
            "  rw [search_succ_eq_scan]",
            f"  change scan edge (search edge 3) 4 [{p}] {p} {prefix}Tail0 {best_term(before)} = {best_term(after)}",
            f"  exact {prefix}Scan0",
        ])
        # Small roots fit in one bounded numeric module. Avoid repeated
        # compiler startups while retaining separate chunks for larger roots.
        if len(numeric) <= STEPS_PER_FILE:
            combined = list(setup)
            for declarations in numeric:
                combined.extend(declarations)
            combined.extend(assembly[1 + len(numeric_modules) + len(HEADER) :])
            return {
                DESTINATION / f"SearchRoot{index:03}.lean": "\n".join([*combined, FOOTER, ""]),
            }, after
        files[DESTINATION / f"SearchRoot{index:03}.lean"] = "\n".join([*assembly, FOOTER, ""])
        return files, after

    def files(self) -> dict[Path, str]:
        setup = [
            "import ProjectEulerSolutions.Lib.PrimePairLarge.FastRows",
            "import ProjectEulerSolutions.Lib.CliqueSearchRows", *HEADER,
            "theorem candidates_append_of_within (relation : Nat → Nat → Bool) (need : Nat) (sum : Int) (p : Nat)",
            "    (left right : List Nat) (bound : Int) (inside : ∀ q ∈ left, sum + (p : Int) + (q : Int) < bound) :",
            "    candidates relation need sum p (left ++ right) bound =",
            "      left.filter (relation p) ++ candidates relation need sum p right bound := by",
            "  unfold candidates",
            "  rw [List.takeWhile_append_of_pos (by intro q member; simpa only [decide_eq_true_eq] using inside q member), List.filter_append]",
            "def edge : Nat → Nat → Bool := Data.graph.lookupEdge",
            "def seed : Best := ⟨26034, []⟩",
            f"def winner : Best := ⟨26033, {list(WINNER[1])}⟩",
            f"def tail{len(self.domain)} : List Nat := []",
            *(f"def tail{i} : List Nat := {self.domain[i]} :: tail{i + 1}" for i in reversed(range(len(self.domain)))),
        ]
        _, _, blocks = large.layout()
        joins = [
            "large", "large_left", "large_right", "large_left_left", "large_left_right",
            "large_left_right_right", "large_right_left", "large_right_right",
            "large_right_right_right", "small", "small_left", "small_right", "small_right_right",
        ]
        block_proofs = []
        for name, lower, upper in blocks:
            start = sum(p < lower for p in self.domain)
            end = sum(p < upper for p in self.domain)
            theorem = f"tail_{name}"
            block_proofs.append(theorem)
            setup.extend([
                "set_option maxRecDepth 4096 in",
                f"theorem {theorem} : tail{start} = PrimePairVertices.{name} ++ tail{end} := rfl",
            ])
        setup.extend([
            "theorem tail0_eq_vertices : tail0 = PrimePairVertices.large := by",
            f"  rw [{', '.join(block_proofs)}, tail{len(self.domain)}]",
            f"  simp only [{', '.join('PrimePairVertices.' + name for name in joins)}, List.append_assoc, List.append_nil]",
        ])
        files = self.fast_row_files()
        files[DESTINATION / "SearchSetup.lean"] = "\n".join([*setup, FOOTER, ""])
        before = SEED
        roots = []
        for i, p in enumerate(self.domain):
            if before[0] <= 5 * p:
                break
            generated, after = self.root_files(i, before)
            files.update(generated)
            roots.append((i, p, before, after))
            before = after
        assert before == WINNER
        stop = len(roots)
        setup.extend([
            "theorem large_ordered : PrimePairVertices.large.Pairwise (· < ·) := by",
            "  rw [← PrimePairVertices.large_checked, PrimePairCertificate.primeSegment]",
            "  exact (List.sortedLT_range' 2 26032 (s := 1) (by decide)).pairwise.sublist List.filter_sublist",
            "theorem tail0_ordered : tail0.Pairwise (· < ·) := by",
            "  rw [tail0_eq_vertices]",
            "  exact large_ordered",
        ])
        for i in range(stop):
            setup.extend([
                f"theorem tail{i + 1}_ordered : tail{i + 1}.Pairwise (· < ·) :=",
                f"  (List.pairwise_cons.mp tail{i}_ordered).2",
            ])
        files[DESTINATION / "SearchSetup.lean"] = "\n".join([*setup, FOOTER, ""])
        final = ["import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup", *HEADER]
        final.extend([
            f"theorem outerScan{stop} : scan edge (search edge 4) 5 [] 0 tail{stop} winner = winner := by",
            f"  rw [tail{stop}, scan, if_pos (by decide)]",
        ])
        files[DESTINATION / "SearchEnd.lean"] = "\n".join([*final, FOOTER, ""])
        previous = "SearchEnd"
        for chunk, end in enumerate(range(stop, 0, -ROOTS_PER_SCAN_FILE)):
            group = roots[max(0, end - ROOTS_PER_SCAN_FILE) : end]
            lines = [
                f"import ProjectEulerSolutions.Lib.PrimePairLarge.{previous}",
                *(f"import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot{i:03}" for i, _, _, _ in group),
                *HEADER,
            ]
            for i, p, prior, after in reversed(group):
                lines.extend([
                    "set_option maxRecDepth 4096 in",
                    f"theorem outerScan{i} : scan edge (search edge 4) 5 [] 0 tail{i} {best_term(prior)} = winner := by",
                    f"  rw [tail{i}, scan, if_neg (by decide), root{i}Candidates_checked]",
                    f"  change scan edge (search edge 4) 5 [] 0 tail{i + 1}",
                    f"    (if 4 ≤ root{i}Candidates.length then search edge 4 [{p}] root{i}Candidates {p} {best_term(prior)} else {best_term(prior)}) = winner",
                    f"  rw [if_pos (by decide), root{i}Branch]",
                    f"  exact outerScan{i + 1}",
                ])
            if group[0][0] == 0:
                lines.extend([
                    "theorem largeSearch : search Data.graph.lookupEdge 5 [] PrimePairVertices.large 0 seed = winner := by",
                    "  rw [search_succ_eq_scan, ← tail0_eq_vertices]",
                    "  exact outerScan0",
                ])
            previous = f"Search{chunk:03}"
            files[DESTINATION / f"{previous}.lean"] = "\n".join([*lines, FOOTER, ""])
        certificate = [
            "import ProjectEulerSolutions.Lib.PrimePairLarge.Graph",
            f"import ProjectEulerSolutions.Lib.PrimePairLarge.{previous}",
            "namespace ProjectEulerSolutions.PrimePairLarge",
            "theorem large_search_checked : CliqueSearch.search graph.lookupEdge 5 [] PrimePairVertices.large 0",
            f"    ⟨26034, []⟩ = ⟨26033, {list(WINNER[1])}⟩ := by",
            "  rw [graph_eq_data]",
            "  exact Search.largeSearch",
            "end ProjectEulerSolutions.PrimePairLarge",
        ]
        files[DESTINATION / "Certificate.lean"] = "\n".join([*certificate, ""])
        return files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare generated files without writing")
    arguments = parser.parse_args()
    generated = Trace(base.prime_values(base.LARGE_END - 1)).files()
    stale = []
    for path, content in generated.items():
        if arguments.check:
            if not path.exists() or path.read_text() != content:
                stale.append(str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists() or path.read_text() != content:
                path.write_text(content)
    if stale:
        raise SystemExit("Outdated seeded-search certificates:\n" + "\n".join(stale))
    print(f"{'Checked' if arguments.check else 'Generated'} {len(generated)} seeded-search P60 certificate files")
