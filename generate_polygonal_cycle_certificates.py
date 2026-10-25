#!/usr/bin/env python3
"""
Generate bounded kernel checks for Euler 61's ordered search models.

Reference roots are checked in independent modules to bound Lean memory use.
The generated Python results are proposals; Lean checks every one.
"""

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DESTINATION = ROOT / "ProjectEulerSolutions/Lib/PolygonalCycleReference"
ROOTS_PER_FILE = 4


def polygonal(sides: int, index: int) -> int:
    return index * ((sides - 2) * index - (sides - 4)) // 2


def reference_results() -> tuple[list[int], list[int]]:
    families = [
        [value for n in range(141) if 1000 <= (value := polygonal(sides, n)) <= 9999]
        for sides in range(3, 9)
    ]

    def extend(path: list[int], remaining: list[int]) -> int:
        if not remaining:
            return sum(path) if path[-1] % 100 == path[0] // 100 else 0
        best = 0
        for kind in remaining:
            unused = [other for other in remaining if other != kind]
            for value in families[kind]:
                if value not in path and path[-1] % 100 == value // 100:
                    best = max(best, extend(path + [value], unused))
        return best

    return families[0], [extend([start], [1, 2, 3, 4, 5]) for start in families[0]]


def certificate_files() -> dict[Path, str]:
    starts, results = reference_results()
    count = (len(starts) + ROOTS_PER_FILE - 1) // ROOTS_PER_FILE
    data = [
        "import ProjectEulerSolutions.Lib.PolygonalCycle",
        "",
        "namespace ProjectEulerSolutions.PolygonalCycle.Reference",
        f"def starts : List Int := {starts}",
        f"def results : List Int := {results}",
        "def visit (start : Int) : Int := extend 140 5 [start] [1, 2, 3, 4, 5]",
        "set_option maxRecDepth 4096 in",
        "theorem starts_checked : starts = fourDigit 140 3 := by decide",
        "end ProjectEulerSolutions.PolygonalCycle.Reference",
    ]
    files = {DESTINATION / "Data.lean": "\n".join(data) + "\n"}
    for part in range(count):
        first = part * ROOTS_PER_FILE
        stop = first + ROOTS_PER_FILE
        lines = [
            "import ProjectEulerSolutions.Lib.PolygonalCycleReference.Data",
            "",
            "/-! Independent reference roots keep the kernel check within its memory cap. -/",
            f"namespace ProjectEulerSolutions.PolygonalCycle.Reference.Part{part:03}",
            f"def starts : List Int := {starts[first:stop]}",
            f"def results : List Int := {results[first:stop]}",
            "set_option maxRecDepth 4096 in",
            "set_option maxHeartbeats 2000000 in",
            "-- Four exhaustive reference roots are evaluated by the kernel together.",
            "theorem checked : starts.map visit = results := by decide",
            f"end ProjectEulerSolutions.PolygonalCycle.Reference.Part{part:03}",
        ]
        files[DESTINATION / f"Part{part:03}.lean"] = "\n".join(lines) + "\n"
    combined_starts = " ++ ".join(f"Part{part:03}.starts" for part in range(count))
    combined_results = " ++ ".join(f"Part{part:03}.results" for part in range(count))
    certificates = ", ".join(f"Part{part:03}.checked" for part in range(count))
    certificate = [
        *(f"import ProjectEulerSolutions.Lib.PolygonalCycleReference.Part{part:03}" for part in range(count)),
        "import ProjectEulerSolutions.Lib.PolygonalCycleOptimized",
        "",
        "namespace ProjectEulerSolutions.PolygonalCycle.Reference",
        "theorem results_checked : starts.map visit = results := by",
        f"  change ({combined_starts}).map visit = {combined_results}",
        f"  simp only [List.map_append, {certificates}]",
        "",
        "private theorem fold_results (values : List Int) (bound : Int) :",
        "    values.foldl (fun best start => max best (visit start)) bound =",
        "      (values.map visit).foldl max bound := by",
        "  induction values generalizing bound with",
        "  | nil => rfl",
        "  | cons value rest ih => exact ih (max bound (visit value))",
        "",
        "theorem reference_checked : reference 140 = 28684 := by",
        "  unfold reference",
        "  rw [← starts_checked]",
        "  change starts.foldl (fun best start => max best (visit start)) 0 = 28684",
        "  rw [fold_results, results_checked]",
        "  decide",
        "",
        "theorem models_equal : answer = reference 140 := by",
        "  rw [answer_checked, reference_checked]",
        "end ProjectEulerSolutions.PolygonalCycle.Reference",
    ]
    files[DESTINATION / "Certificate.lean"] = "\n".join(certificate) + "\n"
    optimized = [
        "import ProjectEulerSolutions.Lib.PolygonalCycle",
        "",
        "namespace ProjectEulerSolutions.PolygonalCycle",
        "set_option maxRecDepth 4096 in",
        "set_option maxHeartbeats 2000000 in",
        "-- The kernel evaluates the complete ordered DFS, including unsuccessful starts.",
        "theorem cycle_checked :",
        "    cycle = some [(1281, 8), (8128, 6), (2882, 5), (8256, 3), (5625, 4), (2512, 7)] := by decide",
        "",
        "theorem answer_checked : answer = 28684 := by",
        "  rw [answer, cycle_checked]",
        "  decide",
        "end ProjectEulerSolutions.PolygonalCycle",
    ]
    files[ROOT / "ProjectEulerSolutions/Lib/PolygonalCycleOptimized.lean"] = "\n".join(optimized) + "\n"
    return files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check generated files without writing")
    arguments = parser.parse_args()
    generated = certificate_files()
    for path, content in generated.items():
        if arguments.check:
            if not path.exists() or path.read_text() != content:
                raise SystemExit(f"Stale certificate: {path.relative_to(ROOT)}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    print(f"{'Checked' if arguments.check else 'Generated'} {len(generated)} bounded Euler 61 certificate files")
