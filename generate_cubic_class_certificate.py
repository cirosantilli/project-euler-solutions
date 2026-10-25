#!/usr/bin/env python3
"""
Generate bounded kernel checks for one complete five-cube digit class.

The generated root lists are proposals; Lean checks their exact completeness.
"""

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DESTINATION = ROOT / "ProjectEulerSolutions/Lib/CubicFiveClass"


def certificate_files() -> dict[Path, str]:
    first_root = 1
    example_root = 5027
    stop = 1
    ceiling = 10 ** len(str(example_root ** 3))
    while stop ** 3 < ceiling:
        stop += 1
    digits = [str(example_root ** 3).count(str(digit)) for digit in range(10)]
    roots = [
        root for root in range(first_root, stop)
        if [str(root ** 3).count(str(digit)) for digit in range(10)] == digits
    ]
    assert roots == [5027, 7061, 7202, 8288, 8384]
    files = {}
    namespace = "ProjectEulerSolutions.CubicFiveClass"
    files[DESTINATION / "Digits.lean"] = (
        "import ProjectEulerSolutions.Lib.CubicClassWitness\n\n"
        "/-! The digit histogram identifying the five-cube existence witness. -/\n"
        f"namespace {namespace}\n"
        "open CubicPermutations CubicSearch\n"
        "set_option Elab.async false\n\n"
        f"def digits : List Nat := {digits}\n\n"
        "theorem example_signature : signature (cube 5027) = digits := by decide\n\n"
        f"end {namespace}\n"
    )

    def join_proof(first: int, count: int, checks: list[tuple[int, str]]) -> str:
        steps = []

        def join(start: int, size: int, parts: list[tuple[int, str]]) -> str:
            if len(parts) == 1:
                return parts[0][1]
            middle = len(parts) // 2
            left = sum(part_size for part_size, _ in parts[:middle])
            left_roots = [root for root in roots if start <= root < start + left]
            right_roots = [root for root in roots if start + left <= root < start + size]
            before = join(start, left, parts[:middle])
            after = join(start + left, size - left, parts[middle:])
            name = f"joined{len(steps)}"
            steps.append(
                f"  have {name} := rootsWith_join digits {start} {left} {size - left}\n"
                f"    {left_roots} {right_roots} {before} {after}"
            )
            return name

        result = join(first, count, checks)
        return "\n".join([*steps, f"  exact {result}"])

    modules = []
    for block, first in enumerate(range(first_root, stop, 1024)):
        name = f"Block{block:03}"
        count = min(1024, stop - first)
        text = [
            "import ProjectEulerSolutions.Lib.CubicFiveClass.Digits\n",
            "/-! Independent small checks bound kernel memory for class completeness. -/",
            f"namespace {namespace}.{name}",
            "open CubicClassWitness",
            "set_option Elab.async false",
            "-- Kernel reduction checks at most 128 root signatures per declaration.",
            "set_option maxRecDepth 4096\n",
        ]
        checks = []
        for index, start in enumerate(range(first, first + count, 128)):
            size = min(128, first + count - start)
            selected = [root for root in roots if start <= root < start + size]
            check = f"part{index}"
            text += [
                "set_option maxHeartbeats 1000000 in",
                "-- Each check reduces at most 128 cube signatures and their digit comparisons.",
                f"theorem {check} : rootsWith digits {start} {size} = {selected} := by decide\n"
            ]
            checks.append((size, check))
        selected = [root for root in roots if first <= root < first + count]
        proof = join_proof(first, count, checks)
        text += [
            f"theorem checked : rootsWith digits {first} {count} = {selected} := by",
            f"{proof}\n",
            f"end {namespace}.{name}\n",
        ]
        files[DESTINATION / f"{name}.lean"] = "\n".join(text)
        modules.append((count, f"{name}.checked"))

    proof = join_proof(first_root, stop - first_root, modules)
    text = [
        *(f"import ProjectEulerSolutions.Lib.CubicFiveClass.Block{i:03}"
          for i in range(len(modules))),
        "\n/-! One complete digit class proves existence without executing either solver. -/",
        f"namespace {namespace}",
        "open CubicClassWitness CubicSearch CubicGroupCounts CubicWindows",
        "set_option Elab.async false\n",
        f"theorem exact_roots : rootsWith digits 1 {stop - 1} = {roots} := by",
        f"{proof}\n",
        "theorem five_count :",
        f"    ((rootsWith (CubicPermutations.signature (cube {example_root})) 1 {stop - 1})"
        ".length : Int) = 5 := by",
        "  rw [example_signature, exact_roots]",
        "  rfl\n",
        f"theorem model_returns : Returns 5 start (choose 5 (collect (prior {stop}))) :=",
        f"  returns_of_count {stop} {example_root} 5"
        " (by decide) (by decide) (by decide) five_count\n",
        f"end {namespace}\n",
    ]
    files[DESTINATION / "Certificate.lean"] = "\n".join(text)
    return files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check without writing files")
    options = parser.parse_args()
    files = certificate_files()
    for path, content in files.items():
        if options.check:
            if not path.exists() or path.read_text() != content:
                raise SystemExit(f"Certificate differs: {path.relative_to(ROOT)}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    print(f"{'Checked' if options.check else 'Wrote'} {len(files)} certificate files")
