#!/usr/bin/env python3
"""
Generate P60's bounded prime-vertex and adjacency certificates.

The producer supplies finite lists. Lean checks every interval against a
balanced divisor classifier and proves equality with genuine prime filtering.
"""

import argparse
from math import isqrt
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DESTINATION = ROOT / "ProjectEulerSolutions/Lib/PrimePairVertices"
NAMESPACE = "ProjectEulerSolutions.PrimePairVertices"
SMALL_END = 8389
LARGE_END = 26034
INTERVAL_SIZE = 128
INTERVALS_PER_FILE = 16
SMALL_CACHE_ROWS = 128
CACHE_ROWS_PER_FILE = 8


def prime_values(limit: int) -> list[int]:
    sieve = bytearray([1]) * (limit + 1)
    sieve[:2] = b"\0\0"
    for p in range(2, isqrt(limit) + 1):
        if sieve[p]:
            sieve[p * p : limit + 1 : p] = b"\0" * ((limit - p * p) // p + 1)
    return [p for p in range(3, limit + 1) if p != 5 and sieve[p]]


def combine(
    lines: list[str], parts: list[tuple[str, int, int]], name: str,
    merges: list[tuple[str, str, str]],
) -> tuple[str, int, int]:
    if len(parts) == 1:
        return parts[0]
    middle = len(parts) // 2
    left, start, split = combine(lines, parts[:middle], name + "_left", merges)
    right, _, end = combine(lines, parts[middle:], name + "_right", merges)
    merges.append((name, left, right))
    lines.extend(
        [
            f"def {name} : List Nat := {left} ++ {right}",
            f"theorem {name}_checked : primeSegment {start} {end - start} = {name} := by",
            f"  change primeSegment {start} ({split - start} + {end - split}) = _",
            f"  rw [primeSegment_append, {left}_checked, {right}_checked]",
            "  rfl",
            "",
        ]
    )
    return name, start, end


def source_prime_check(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def source_pair_check(p: int, q: int) -> bool:
    return (
        (p + q) % 3 != 0
        and source_prime_check(int(str(p) + str(q)))
        and source_prime_check(int(str(q) + str(p)))
    )


def row_file(
    p: int, intervals: list[tuple[int, int]], primes: list[int],
    merges: list[tuple[str, str, str]],
) -> str:
    lines = [
        "import ProjectEulerSolutions.Lib.PrimePairVertices.All",
        "",
        "/-! Generated adjacency data; each source-model test is kernel checked. -/",
        "namespace ProjectEulerSolutions.PrimePairRows",
        "open PrimePairCertificate",
        "",
    ]
    for start, end in intervals:
        name = f"interval_{start}"
        accepted = [q for q in primes if start <= q < end and source_pair_check(p, q)]
        lines.extend([
            f"def p{p}_{name} : List Nat := {accepted}",
            f"theorem p{p}_{name}_checked : PrimePairVertices.{name}.filter (relation {p}) = p{p}_{name} := by decide",
            "",
        ])
    for name, left, right in merges:
        lines.extend([
            f"def p{p}_{name} : List Nat := p{p}_{left} ++ p{p}_{right}",
            f"theorem p{p}_{name}_checked : PrimePairVertices.{name}.filter (relation {p}) = p{p}_{name} := by",
            f"  rw [PrimePairVertices.{name}, List.filter_append, p{p}_{left}_checked, p{p}_{right}_checked]",
            "  rfl",
            "",
        ])
    lines.append("end ProjectEulerSolutions.PrimePairRows")
    return "\n".join(lines) + "\n"


def small_first_branch(primes: list[int]) -> dict[Path, str]:
    """Split the first bounded search branch into short checked scan steps."""
    values = [q for q in primes if q < SMALL_END and source_pair_check(3, q)]
    size = len(values)
    output = {}
    for chunk, stop in enumerate(range(size, 0, -32)):
        start = max(0, stop - 32)
        lines = [
            f"import ProjectEulerSolutions.Lib.PrimePairSearch.SmallP3Chunk{chunk - 1:03}"
            if chunk else "import ProjectEulerSolutions.Lib.PrimePairRows.P3",
            "import ProjectEulerSolutions.Lib.CliqueSearch",
            "",
            "/-! Generated scan steps for the first small-domain branch. -/",
            "namespace ProjectEulerSolutions.PrimePairSearch",
            "open PrimePairCertificate CliqueSearch",
            "",
        ]
        if chunk == 0:
            lines.extend([
                "def smallBest : Best := ⟨10 ^ 30, []⟩",
                f"def smallTail{size} : List Nat := []",
                f"theorem smallScan{size} : scan relation (search relation 3) 4 [3] 3 smallTail{size} smallBest = smallBest := rfl",
                "",
            ])
        for i in reversed(range(start, stop)):
            q = values[i]
            candidates = [r for r in values[i + 1 :] if source_pair_check(q, r)]
            lines.extend([
                f"def smallTail{i} : List Nat := {q} :: smallTail{i + 1}",
                f"def smallCandidates{i} : List Nat := {candidates}",
                "set_option maxRecDepth 2048 in",
                f"theorem smallCandidates{i}_checked : candidates relation 4 3 {q} smallTail{i + 1} smallBest.bound = smallCandidates{i} := by decide",
                "",
            ])
            if len(candidates) >= 3:
                lines.extend([
                    "set_option maxRecDepth 2048 in",
                    f"theorem smallChild{i} : search relation 3 [3, {q}] smallCandidates{i} {3 + q} smallBest = smallBest := by decide",
                    "",
                ])
            lines.extend([
                f"theorem smallScan{i} : scan relation (search relation 3) 4 [3] 3 smallTail{i} smallBest = smallBest := by",
                f"  rw [smallTail{i}, scan, if_neg (by decide), smallCandidates{i}_checked]",
                f"  change scan relation (search relation 3) 4 [3] 3 smallTail{i + 1}",
                f"    (if 3 ≤ smallCandidates{i}.length then search relation 3 [3, {q}] smallCandidates{i} {3 + q} smallBest else smallBest) = smallBest",
                f"  rw [if_pos (by decide), smallChild{i}]"
                if len(candidates) >= 3 else "  rw [if_neg (by decide)]",
                f"  exact smallScan{i + 1}",
                "",
            ])
        if start == 0:
            lines.extend([
                "theorem small_p3_branch : search relation 4 [3] PrimePairRows.p3_small 3 smallBest = smallBest := by",
                "  rw [search_succ_eq_scan]",
                "  change scan relation (search relation 3) 4 [3] 3 smallTail0 smallBest = smallBest",
                "  exact smallScan0",
                "",
            ])
        lines.append("end ProjectEulerSolutions.PrimePairSearch")
        output[ROOT / f"ProjectEulerSolutions/Lib/PrimePairSearch/SmallP3Chunk{chunk:03}.lean"] = "\n".join(lines) + "\n"
    return output


def vertex_tree_file(
    intervals: list[tuple[int, int]], merges: list[tuple[str, str, str]],
) -> str:
    lines = [
        "import ProjectEulerSolutions.Lib.PrimePairVertices.All",
        "import ProjectEulerSolutions.Lib.PrimePairGraph",
        "",
        "/-! Balanced views of the already certified prime lists. -/",
        f"namespace {NAMESPACE}",
        "open PrimePairCertificate",
        "",
    ]
    for start, _ in intervals:
        lines.append(f"def tree_interval_{start} : VertexTree := .leaf interval_{start}")
    for name, left, right in merges:
        lines.append(f"def tree_{name} : VertexTree := .branch tree_{left} tree_{right}")
    lines.extend([
        "",
        "def smallTree : VertexTree := tree_small",
        "theorem smallTree_values : smallTree.values = small := rfl",
        "def largeTree : VertexTree := tree_large",
        "theorem largeTree_values : largeTree.values = large := rfl",
        "",
        f"end {NAMESPACE}",
    ])
    return "\n".join(lines) + "\n"


def cached_small_rows(
    primes: list[int], blocks: list[tuple[str, int, int]],
    merges: list[tuple[str, str, str]],
) -> dict[Path, str]:
    """Certify each increasing row once, then assemble a binary lookup cache."""
    domain = [p for p in primes if p < SMALL_END]
    selected = domain[:SMALL_CACHE_ROWS]
    output = {}
    groups = [selected[i : i + CACHE_ROWS_PER_FILE] for i in range(0, len(selected), CACHE_ROWS_PER_FILE)]
    for index, group in enumerate(groups):
        lines = [
            "import ProjectEulerSolutions.Lib.PrimePairVertices.Trees",
            "",
            "/-! Increasing adjacency rows, checked against the exact source predicate. -/",
            "namespace ProjectEulerSolutions.PrimePairCache",
            "open PrimePairCertificate",
            "",
        ]
        for p in group:
            for name, start, end in blocks:
                neighbors = [q for q in domain if start <= q < end and p < q and source_pair_check(p, q)]
                lines.extend([
                    f"def smallRow{p}_{name} : List Nat := {neighbors}",
                    "set_option maxRecDepth 2048 in",
                    f"theorem smallRow{p}_{name}_checked : upperNeighbors PrimePairVertices.tree_{name} {p} = smallRow{p}_{name} := by decide",
                    "",
                ])
            for name, left, right in merges:
                lines.extend([
                    f"def smallRow{p}_{name} : List Nat := smallRow{p}_{left} ++ smallRow{p}_{right}",
                    f"theorem smallRow{p}_{name}_checked : upperNeighbors PrimePairVertices.tree_{name} {p} = smallRow{p}_{name} := by",
                    f"  change upperNeighbors PrimePairVertices.tree_{left} {p} ++ upperNeighbors PrimePairVertices.tree_{right} {p} = _",
                    f"  rw [smallRow{p}_{left}_checked, smallRow{p}_{right}_checked]",
                    "  rfl",
                    "",
                ])
            lines.extend([
                f"def smallRow{p} : List Nat := smallRow{p}_small",
                f"theorem smallRow{p}_checked : upperNeighbors PrimePairVertices.smallTree {p} = smallRow{p} := smallRow{p}_small_checked",
                "",
            ])
        lines.append("end ProjectEulerSolutions.PrimePairCache")
        output[ROOT / f"ProjectEulerSolutions/Lib/PrimePairCache/SmallRows{index:03}.lean"] = "\n".join(lines) + "\n"
    lines = [
        *(f"import ProjectEulerSolutions.Lib.PrimePairCache.SmallRows{i:03}" for i in range(len(groups))),
        "",
        "namespace ProjectEulerSolutions.PrimePairCache",
        "open PrimePairCertificate",
        "",
        "def smallEmpty : Graph := .empty",
        "theorem smallEmpty_agrees : Graph.Agrees PrimePairVertices.smallTree smallEmpty := Graph.empty_agrees _",
        "",
    ]

    def assemble(values: list[int]) -> str:
        if not values:
            return "smallEmpty"
        middle = len(values) // 2
        p = values[middle]
        left = assemble(values[:middle])
        right = assemble(values[middle + 1 :])
        name = f"smallNode{p}"
        lines.extend([
            f"def {name} : Graph := .node {p} smallRow{p} {left} {right}",
            "set_option maxRecDepth 4096 in",
            f"theorem {name}_agrees : Graph.Agrees PrimePairVertices.smallTree {name} :=",
            f"  Graph.node_agrees _ {p} smallRow{p} {left} {right}",
            f"    {left}_agrees {right}_agrees (by decide) (by decide) smallRow{p}_checked",
            "",
        ])
        return name

    name = assemble(selected)
    lines.extend([
        f"def smallPrefix : Graph := {name}",
        f"theorem smallPrefix_agrees : Graph.Agrees PrimePairVertices.smallTree smallPrefix := {name}_agrees",
        "set_option maxRecDepth 4096 in",
        f"theorem smallPrefix_vertices : smallPrefix.vertices = PrimePairVertices.small.take {len(selected)} := rfl",
        "",
        "theorem smallPrefix_vertices_subset (p : Nat) (member : p ∈ smallPrefix.vertices) : p ∈ PrimePairVertices.small := by",
        "  rw [smallPrefix_vertices] at member",
        "  exact List.mem_of_mem_take member",
        "",
        "end ProjectEulerSolutions.PrimePairCache",
    ])
    output[ROOT / "ProjectEulerSolutions/Lib/PrimePairCache/SmallPrefix.lean"] = "\n".join(lines) + "\n"
    return output


def generated_files() -> dict[Path, str]:
    primes = prime_values(LARGE_END - 1)
    boundaries = sorted(set(range(2, LARGE_END, INTERVAL_SIZE)) | {SMALL_END, LARGE_END})
    intervals = list(zip(boundaries, boundaries[1:]))
    groups = []
    for selected in (
        [interval for interval in intervals if interval[1] <= SMALL_END],
        [interval for interval in intervals if interval[0] >= SMALL_END],
    ):
        groups.extend(
            selected[i : i + INTERVALS_PER_FILE]
            for i in range(0, len(selected), INTERVALS_PER_FILE)
        )
    output = {}
    roots = []
    merges = []
    for index, group in enumerate(groups):
        lines = [
            "import ProjectEulerSolutions.Lib.PrimePairCertificate",
            "",
            "/-! Generated bounded vertex intervals; every list is kernel checked. -/",
            f"namespace {NAMESPACE}",
            "open PrimePairCertificate",
            "",
        ]
        parts = []
        for start, end in group:
            name = f"interval_{start}"
            values = [p for p in primes if start <= p < end]
            lines.extend(
                [
                    f"def {name} : List Nat := {values}",
                    f"theorem {name}_checked : primeSegment {start} {end - start} = {name} := by",
                    f"  rw [← computedSegment_eq_primeSegment {start} {end - start} (by decide)]",
                    "  decide",
                    "",
                ]
            )
            parts.append((name, start, end))
        roots.append(combine(lines, parts, f"block_{index:03}", merges))
        lines.append(f"end {NAMESPACE}")
        output[DESTINATION / f"Chunk{index:03}.lean"] = "\n".join(lines) + "\n"
    lines = [
        *(f"import ProjectEulerSolutions.Lib.PrimePairVertices.Chunk{i:03}" for i in range(len(groups))),
        "",
        f"namespace {NAMESPACE}",
        "open PrimePairCertificate",
        "",
    ]
    small = combine(lines, [part for part in roots if part[2] <= SMALL_END], "small", merges)
    large = combine(lines, [small, *(part for part in roots if part[1] >= SMALL_END)], "large", merges)
    assert small == ("small", 2, SMALL_END)
    assert large == ("large", 2, LARGE_END)
    lines.append(f"end {NAMESPACE}")
    output[DESTINATION / "All.lean"] = "\n".join(lines) + "\n"
    output[ROOT / "ProjectEulerSolutions/Lib/PrimePairRows/P3.lean"] = row_file(3, intervals, primes, merges)
    output.update(small_first_branch(primes))
    output[DESTINATION / "Trees.lean"] = vertex_tree_file(intervals, merges)
    output.update(cached_small_rows(
        primes, [part for part in roots if part[2] <= SMALL_END],
        [merge for merge in merges if merge[0] == "small" or merge[0].startswith("small_")],
    ))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare without writing")
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
        raise SystemExit("Outdated generated certificates:\n" + "\n".join(stale))
    print(f"{'Checked' if arguments.check else 'Generated'} {len(files)} P60 certificate files")
