#!/usr/bin/env python3
"""
Regenerate source ASTs and check runtime and exploratory cost proofs serially.
Uses the project's installed Lake dependencies; does not run Python solvers.
"""

import argparse
from dataclasses import replace
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import tempfile

if __package__:
    from .lean_checked import run_lean
else:
    from lean_checked import run_lean

ROOT = Path(__file__).resolve().parents[1]
FORMALSW = ROOT / "data/project-euler-statements/formalsw"


def public_theorems(source):
    """Collect the explicit namespace/theorem declarations in these proof files."""
    namespaces = []
    for line in source.read_text().splitlines():
        line = re.sub(r"^(?:@\[[^]]*\]\s*)+", "", line.strip())
        if line.startswith("namespace "):
            namespaces.append(line.split(maxsplit=1)[1])
        elif line.startswith("end "):
            namespaces.pop()
        elif match := re.match(r"(?:protected\s+)?(?:theorem|lemma)\s+((?:«[^»]*»|[^\s(«])+)", line):
            yield ".".join([*namespaces, match[1]])


def proof_path(problem):
    name = "RuntimeCost" if problem > 10 else "MarkingWork" if problem in (7, 10) else "Complexity"
    return Path(f"solvers/{problem}.py.lean.d/{name}.lean")


def runtime_proofs(problem):
    for source in (Path(f"solvers/{problem}.py"), Path(
            f"data/project-euler-statements/data/py/project_euler_statements/p{problem}.py")):
        proof = source.with_name(source.name + ".lean.d") / "RuntimeCost.lean"
        if not proof.is_file():
            raise RuntimeError(f"Missing runtime proof: {proof}")
        yield source, proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problems", type=int, nargs="*", default=list(range(1, 11)), choices=range(1, 21))
    args = parser.parse_args()
    problems = sorted(set(args.problems))
    os.chdir(ROOT)
    sys.path.insert(0, str(FORMALSW))
    sys.path.insert(0, str(ROOT / "nagini/src"))
    frontend = runpy.run_path(str(FORMALSW / "formalsw"))
    environment = dict(os.environ, LEAN_NUM_THREADS="1")
    lean_path = subprocess.check_output(
        ["lake", "env", "printenv", "LEAN_PATH"], text=True, env=environment
    ).strip()
    build = ROOT / ".lake/build/lib/lean"
    environment["LEAN_PATH"] = str(build) + os.pathsep + lean_path

    checked = set()

    def check(source, output=None, source_root=None):
        key = (source.resolve(), output, source_root)
        if key in checked:
            return
        # lake env resets LEAN_PATH, so invoke its selected Lean directly instead.
        command = [lean, "-DwarningAsError=true"]
        if source_root is not None:
            command.append(f"--root={source_root}")
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            command += ["-o", str(output)]
        command.append(str(source))
        print(f"Checking {source.relative_to(ROOT)}", flush=True)
        run_lean(command, env=environment, check=True, timeout=300)
        checked.add(key)

    lean = subprocess.check_output(
        ["lake", "env", "which", "lean"], text=True, env=environment
    ).strip()
    for name in ("ExecutionCost", "ExecutionCostDomain", "CallsCost", "CallsCostLoops", "CallsCostControl", "HeapCallsCost", "HeapCallsCostLists", "ScalarSortCost", "WeightedExecutionCost", "AllocationCost", "HeapReadCost", "BufferCost", "BufferCostSlice", "RuntimeCost", "PrimitiveWork", "PrimitiveWorkBounds", "CallsRuntimeCost", "CallsRuntimeCostLoops", "CallsRuntimeCostControl", "CallsRuntimeCostFunctions", "CallsRuntimeCostScaled", "CallsRuntimeCostBitBounds", "CallsRuntimeCostNumberBounds", "CallsRuntimeCostLinking", "CallsRuntimeCostBlocks", "DecimalWorkBounds", "IntegerListWork", "FactorialWorkBounds", "CallsRuntimeCostStrings", "CallsRuntimeCostRanges", "LcmRuntimeCost", "FibonacciRuntimeCost", "HeapMapWork", "HeapRecordWork", "HeapSetWork", "HeapSortWork", "HeapBufferWork", "HeapSequenceWork", "DynamicArrayWork", "AmortizedRuntimeCost", "HeapRuntimeCost", "HeapRuntimeInputs", "HeapRuntimeCostLists", "HeapRuntimeCostControl", "HeapRuntimeCostRecords", "HeapRuntimeCostLibrary", "HeapRuntimeCostRanges", "HeapRuntimeCostBuffers", "HeapRuntimeCostBufferLoops", "HeapRuntimeCostReturns", "HeapRuntimeCostPrimitives", "HeapRuntimeCostSequences", "HeapRuntimeCostStrings", "HeapRuntimeCostComprehensions", "HeapRuntimeCostLoopControl", "HeapRuntimeCostListCopies", "HeapRuntimeCostSetup", "HeapRuntimeCostListLoops", "HeapRuntimeCostSumGenerators", "HeapRuntimeCostRangeSums", "HeapRuntimeCostSums", "DecimalSumRuntimeCost", "HeapRuntimeCostSets", "HeapRuntimeCostComparisons", "HeapRuntimeCostSorting", "HeapRuntimeCostMembership", "HeapRuntimeCostBooleans"):
        check(FORMALSW / f"lang/py/FormalSW/{name}.lean",
              FORMALSW / f".lake/build/lib/lean/lang/py/FormalSW/{name}.olean")
    if 18 in problems:
        statements_lean = ROOT / "data/project-euler-statements/data/lean"
        for relative in (Path("ProjectEulerStatements/P18.lean"),
                         Path("ProjectEulerStatements/Lib/TrianglePaths.lean")):
            check(statements_lean / relative,
                  statements_lean / ".lake/build/lib/lean" / relative.with_suffix(".olean"),
                  source_root=statements_lean)
    if 19 in problems:
        statements_lean = ROOT / "data/project-euler-statements/data/lean"
        for relative in (Path("ProjectEulerStatements/P19.lean"),
                         Path("ProjectEulerStatements/Lib/Calendar.lean")):
            check(statements_lean / relative,
                  statements_lean / ".lake/build/lib/lean" / relative.with_suffix(".olean"),
                  source_root=statements_lean)
    if 3 in problems:
        statements_lean = ROOT / "data/project-euler-statements/data/lean"
        relative = Path("ProjectEulerStatements/Lib/DivisorBounds.lean")
        check(statements_lean / relative,
              statements_lean / ".lake/build/lib/lean" / relative.with_suffix(".olean"),
              source_root=statements_lean)
    if set(problems) & {7, 10}:
        statements_lean = ROOT / "data/project-euler-statements/data/lean"
        relative = Path("ProjectEulerStatements/Lib/ReciprocalSum.lean")
        check(statements_lean / relative,
              statements_lean / ".lake/build/lib/lean" / relative.with_suffix(".olean"),
              source_root=statements_lean)
    if 7 in problems:
        statements_lean = ROOT / "data/project-euler-statements/data/lean"
        for relative in (Path("ProjectEulerStatements/P7.lean"),
                         Path("ProjectEulerStatements/Lib/NthPrimeBounds.lean")):
            check(statements_lean / relative,
                  statements_lean / ".lake/build/lib/lean" / relative.with_suffix(".olean"),
                  source_root=statements_lean)
        for source in (Path("ProjectEulerSolutions/P7.lean"),
                       Path("ProjectEulerSolutions/Lib/PrimeSieve.lean"), Path("ProjectEulerSolutions/Equiv/P7.lean")):
            check(source.resolve(), build / source.with_suffix(".olean"))
    sources = [path for problem in problems for path in (
        f"solvers/{problem}.py",
        f"data/project-euler-statements/data/py/project_euler_statements/p{problem}.py",
    )]
    if set(problems) & {3, 4, 7, 10, 13, 16, 20}:
        sources.append("data/project-euler-statements/data/py/project_euler_statements/common.py")
    for relative in sources:
        source = Path(relative)
        module = frontend["source_module"](source)
        namespace = frontend["namespace_for_source"](source)
        frontend["collect_requirements"](
            source, module, frontend["proof_namespace_for_source"](source), namespace
        )
        semantic = frontend["module_with_python_builtins"](source, module)
        bindings = frontend["source_bindings"](source, module, semantic)
        json_path = frontend["ast_json_path_for_source"](source)
        ast_path = frontend["ast_path_for_source"](source)
        ast_path.parent.mkdir(parents=True, exist_ok=True)
        frontend["write_generated"](
            json_path, frontend["AstJsonEmitter"](semantic, bindings_module=bindings).emit()
        )
        frontend["write_generated"](
            ast_path, frontend["AstLeanBridgeEmitter"](
                json_path, namespace, bindings, module.formalsw_conditions
            ).emit()
        )
        check(ast_path.resolve(), frontend["olean_path_for_lean"](ast_path))
    check(FORMALSW / "tests/py/execution_cost.lean")
    check(FORMALSW / "tests/py/runtime_cost.lean")
    check(FORMALSW / "tests/py/runtime_sums.lean")
    check(FORMALSW / "tests/py/runtime_comprehensions.lean")
    check(FORMALSW / "tests/py/runtime_strings.lean")
    check(FORMALSW / "tests/py/runtime_membership.lean")
    check(FORMALSW / "tests/py/runtime_records.lean")
    check(FORMALSW / "tests/py/runtime_range_sums.lean")
    with tempfile.TemporaryDirectory(prefix="formalsw-runtime-inputs-") as directory:
        generated = Path(directory) / "ListInputs.lean"
        subprocess.run([sys.executable, str(FORMALSW / "tests/py/execution_complexity_checks"),
                        "--lean-output", str(generated)], check=True)
        run_lean([lean, "-DwarningAsError=true", str(generated)], env=environment, check=True)
    subprocess.run([sys.executable, str(FORMALSW / "tests/py/nagini_erasure_checks")], check=True)
    if set(problems) & {7, 10}:
        source = Path("data/project-euler-statements/data/py/project_euler_statements/common.py.lean")
        check(source.resolve(), frontend["olean_path_for_lean"](source))
    if set(problems) & {3, 7, 10}:
        source = Path("data/project-euler-statements/data/py/project_euler_statements/common.py.lean.d/PrimeRuntimeCost.lean")
        check(source.resolve(), build / source.with_suffix(".olean"))
    if 7 in problems:
        source = Path("data/project-euler-statements/data/py/project_euler_statements/common.py.lean.d/PrimeRuntimeResult.lean")
        check(source.resolve(), build / source.with_suffix(".olean"))
    if set(problems) & {4, 13, 16, 20}:
        source = Path("data/project-euler-statements/data/py/project_euler_statements/common.py.lean.d/DigitsRuntimeCost.lean")
        check(source.resolve(), build / source.with_suffix(".olean"))
    if set(problems) & {16, 20}:
        source = Path("data/project-euler-statements/data/py/project_euler_statements/common.py.lean.d/DigitSumRuntimeCost.lean")
        check(source.resolve(), build / source.with_suffix(".olean"))
    if 4 in problems:
        source = Path("data/project-euler-statements/data/py/project_euler_statements/p4.py.lean.d/RuntimeDigits.lean")
        check(source.resolve(), build / source.with_suffix(".olean"))
    if 7 in problems:
        reference = Path("data/project-euler-statements/data/py/project_euler_statements")
        for source in (reference / "p7.py.lean.d/RuntimeCost.lean", reference / "p7.py.lean"):
            check(source.resolve(), build / source.with_suffix(".olean"))
        for name in ("Source", "MarkingWork", "MarkingRuntime", "SieveRuntime", "ScanRuntime", "SetupRuntime", "RetryRuntime", "InitialRuntime"):
            source = Path(f"solvers/7.py.lean.d/{name}.lean")
            check(source.resolve(), build / source.with_suffix(".olean"))
    if 10 in problems:
        reference = Path("data/project-euler-statements/data/py/project_euler_statements")
        for source in (reference / "p10.py.lean.d/RuntimeCost.lean", reference / "p10.py.lean"):
            check(source.resolve(), build / source.with_suffix(".olean"))
        for name in ("Source", "MarkingWork", "MarkingRuntime", "SieveRuntime", "SumRuntime", "SetupRuntime"):
            source = Path(f"solvers/10.py.lean.d/{name}.lean")
            check(source.resolve(), build / source.with_suffix(".olean"))
    if 8 in problems:
        for name in ("Source", "FoldRuntime", "SliceRuntime", "WindowRuntime"):
            source = Path(f"data/project-euler-statements/data/py/project_euler_statements/p8.py.lean.d/{name}.lean")
            check(source.resolve(), build / source.with_suffix(".olean"))
        source = Path("ProjectEulerSolutions/Equiv/P8.lean")
        check(source.resolve(), build / source.with_suffix(".olean"))
        for name in ("Source", "ScanRuntime", "SegmentRuntime", "OuterRuntime"):
            source = Path(f"solvers/8.py.lean.d/{name}.lean")
            check(source.resolve(), build / source.with_suffix(".olean"))
    if 9 in problems:
        for name in ("Source", "InnerRuntime", "LoopRuntime"):
            source = Path(f"data/project-euler-statements/data/py/project_euler_statements/p9.py.lean.d/{name}.lean")
            check(source.resolve(), build / source.with_suffix(".olean"))
        for name in ("Source", "SortingRuntime", "ScalarRuntime", "GuardRuntime", "CandidateRuntime", "RangeRuntime", "OuterRuntime", "TailRuntime"):
            source = Path(f"solvers/9.py.lean.d/{name}.lean")
            check(source.resolve(), build / source.with_suffix(".olean"))
        for source in (Path("ProjectEulerSolutions/P9.lean"), Path("ProjectEulerSolutions/Equiv/P9.lean")):
            check(source.resolve(), build / source.with_suffix(".olean"))
    if 13 in problems:
        for name in ("DigitsBERuntimeCost", "OfDigitsRuntimeCost", "OfDigitsBERuntimeCost"):
            source = Path(f"data/project-euler-statements/data/py/project_euler_statements/common.py.lean.d/{name}.lean")
            check(source.resolve(), build / source.with_suffix(".olean"))
        source = Path("data/project-euler-statements/data/py/project_euler_statements/p13.py.lean.d/SourceRuntime.lean")
        check(source.resolve(), build / source.with_suffix(".olean"))
        for name in ("FormattingRuntime", "ParsingRuntime", "FirstDigitsRuntime", "LeadingRuntime", "SourceRuntime"):
            source = Path(f"solvers/13.py.lean.d/{name}.lean")
            check(source.resolve(), build / source.with_suffix(".olean"))
    if 19 in problems:
        for name in ("CalendarRuntime", "MonthLengthRuntime", "CalendarSumsRuntime", "DateRuntime", "CountingMonthRuntime", "CountingLoopsRuntime", "SourceRuntime"):
            source = Path(f"data/project-euler-statements/data/py/project_euler_statements/p19.py.lean.d/{name}.lean")
            check(source.resolve(), build / source.with_suffix(".olean"))
        for name in ("CalendarRuntime", "MonthRuntime", "SourceRuntime"):
            source = Path(f"solvers/19.py.lean.d/{name}.lean")
            check(source.resolve(), build / source.with_suffix(".olean"))
    if 18 in problems:
        for name in ("RowHeap", "RowRuntime", "HelperRuntime", "TriangleRuntime", "RuntimeCost"):
            source = Path(f"data/project-euler-statements/data/py/project_euler_statements/p18.py.lean.d/{name}.lean")
            check(source.resolve(), build / source.with_suffix(".olean"))
        for name in ("RowRuntime", "TriangleRuntime", "RuntimeCost"):
            source = Path(f"solvers/18.py.lean.d/{name}.lean")
            check(source.resolve(), build / source.with_suffix(".olean"))
        # The runtime theorem reuses the functional proof of the shape precondition.
        for source in (Path("data/project-euler-statements/data/py/project_euler_statements/p18.py.lean"),
                       Path("solvers/18.py.lean")):
            check(source.resolve(), frontend["olean_path_for_lean"](source))
    for source, runtime in (pair for problem in problems for pair in runtime_proofs(problem)):
        check(runtime.resolve(), build / runtime.with_suffix(".olean"))
        module = frontend["source_module"](source)
        requirements = [requirement for requirement in frontend["collect_requirements"](
            source, module, frontend["proof_namespace_for_source"](source),
            frontend["namespace_for_source"](source)) if requirement.marker.startswith("BigO(")]
        if not requirements or any(requirement.unsupported for requirement in requirements):
            raise RuntimeError(f"Missing or unsupported runtime BigO declarations: {source}")
        wrapper = ROOT / ".formalsw" / source.with_suffix(".runtime-requirements.lean")
        wrapper.parent.mkdir(parents=True, exist_ok=True)
        # Check each exact source proposition against its declared theorem.
        declaration = source.with_name(source.name + ".lean") if source == Path("solvers/18.py") else runtime
        wrapper.write_text(frontend["RequirementEmitter"](
            declaration, frontend["lean_module_for_path"](declaration),
            [replace(requirement, dependencies=()) for requirement in requirements]).emit())
        check(wrapper)
    for problem in sorted(set(problems) & ({7, 8, 9, 10} | set(range(11, 21)))):
        source = Path(f"data/project-euler-statements/data/py/project_euler_statements/p{problem}.py.lean")
        check(source.resolve(), frontend["olean_path_for_lean"](source))
        source = Path(f"solvers/{problem}.py.lean")
        check(source.resolve(), frontend["olean_path_for_lean"](source))
    if 8 in problems:
        source = Path("solvers/8.py.lean.d/Source.lean")
        check(source.resolve(), build / source.with_suffix(".olean"))
        source = Path("solvers/8.py.lean.d/GeneralCost.lean")
        check(source.resolve(), build / source.with_suffix(".olean"))
        source = Path("solvers/8.py.lean.d/ZeroComplexity.lean")
        check(source.resolve(), build / source.with_suffix(".olean"))
    for problem in problems:
        source = proof_path(problem)
        check(source.resolve(), build / source.with_suffix(".olean"))
    if 10 in problems:
        source = Path("solvers/10.py.lean.d/MarkingRuntime.lean")
        check(source.resolve(), build / source.with_suffix(".olean"))
    audited = 0
    with tempfile.TemporaryDirectory(prefix="formalsw-cost-axioms-") as directory:
        audit = Path(directory) / "Audit.lean"
        def audit_group(label, imports, names):
            audit.write_text(imports + "".join(f"#print axioms {name}\n" for name in names))
            result = run_lean(
                [lean, str(audit)],
                env=environment, text=True, capture_output=True, check=True, timeout=120,
            )
            dependencies = re.findall(r"depends on axioms:\s*\[([^]]*)\]", result.stdout)
            independent = result.stdout.count("does not depend on any axioms")
            if len(dependencies) + independent != len(names):
                raise RuntimeError(f"Incomplete axiom audit for {label}: {result.stdout}")
            for dependency in dependencies:
                unexpected = set(dependency.replace("\n", " ").replace(",", " ").split()) - {
                    "propext", "Classical.choice", "Quot.sound",
                }
                if unexpected:
                    raise RuntimeError(f"Unexpected proof axioms for {label}: {unexpected}")
            print(f"Axiom audit passed for {label}: {len(names)} theorems", flush=True)
            return len(names)

        for problem in problems:
            sources = [proof_path(problem)]
            if problem == 7:
                sources.extend(Path(f"solvers/7.py.lean.d/{name}.lean") for name in
                               ("MarkingRuntime", "SieveRuntime", "ScanRuntime", "SetupRuntime", "RetryRuntime", "InitialRuntime"))
            if problem == 10:
                sources.extend(Path(f"solvers/10.py.lean.d/{name}.lean") for name in
                               ("MarkingRuntime", "SieveRuntime", "SumRuntime", "SetupRuntime"))
            if problem in (3, 7, 10):
                sources.append(Path("data/project-euler-statements/data/py/project_euler_statements/common.py.lean.d/PrimeRuntimeCost.lean"))
            if problem == 7:
                sources.append(Path("data/project-euler-statements/data/py/project_euler_statements/common.py.lean.d/PrimeRuntimeResult.lean"))
            if problem == 8:
                sources.extend(Path(f"solvers/8.py.lean.d/{name}.lean") for name in
                               ("ZeroComplexity", "ScanRuntime", "SegmentRuntime", "OuterRuntime"))
                sources.extend(Path(f"data/project-euler-statements/data/py/project_euler_statements/p8.py.lean.d/{name}.lean")
                               for name in ("FoldRuntime", "SliceRuntime", "WindowRuntime"))
            if problem == 9:
                sources.extend(Path(f"data/project-euler-statements/data/py/project_euler_statements/p9.py.lean.d/{name}.lean")
                               for name in ("Source", "InnerRuntime", "LoopRuntime"))
                sources.extend(Path(f"solvers/9.py.lean.d/{name}.lean") for name in
                               ("SortingRuntime", "ScalarRuntime", "GuardRuntime", "CandidateRuntime", "RangeRuntime", "OuterRuntime", "TailRuntime"))
            if problem in (4, 13, 16, 20):
                sources.append(Path("data/project-euler-statements/data/py/project_euler_statements/common.py.lean.d/DigitsRuntimeCost.lean"))
            if problem in (16, 20):
                sources.append(Path("data/project-euler-statements/data/py/project_euler_statements/common.py.lean.d/DigitSumRuntimeCost.lean"))
            if problem == 13:
                sources.extend(Path(f"data/project-euler-statements/data/py/project_euler_statements/common.py.lean.d/{name}.lean") for name in
                               ("DigitsBERuntimeCost", "OfDigitsRuntimeCost", "OfDigitsBERuntimeCost"))
                sources.append(Path("data/project-euler-statements/data/py/project_euler_statements/p13.py.lean.d/SourceRuntime.lean"))
                sources.extend(Path(f"solvers/13.py.lean.d/{name}.lean") for name in
                               ("FormattingRuntime", "ParsingRuntime", "FirstDigitsRuntime", "LeadingRuntime", "SourceRuntime"))
            if problem == 19:
                sources.extend(Path(f"data/project-euler-statements/data/py/project_euler_statements/p19.py.lean.d/{name}.lean") for name in
                               ("CalendarRuntime", "MonthLengthRuntime", "CalendarSumsRuntime", "DateRuntime", "CountingMonthRuntime", "CountingLoopsRuntime", "SourceRuntime"))
                sources.extend(Path(f"solvers/19.py.lean.d/{name}.lean") for name in
                               ("CalendarRuntime", "MonthRuntime", "SourceRuntime"))
            if problem == 4:
                sources.append(Path("data/project-euler-statements/data/py/project_euler_statements/p4.py.lean.d/RuntimeDigits.lean"))
            sources.extend(runtime for _, runtime in runtime_proofs(problem))
            if problem == 18:
                sources.extend(Path(f"data/project-euler-statements/data/py/project_euler_statements/p18.py.lean.d/{name}.lean")
                               for name in ("RowHeap", "RowRuntime", "HelperRuntime", "TriangleRuntime"))
                sources.extend(Path(f"solvers/18.py.lean.d/{name}.lean") for name in ("RowRuntime", "TriangleRuntime"))
                sources.append(Path("solvers/18.py.lean"))
            sources = list(dict.fromkeys(sources))
            names = [name for source in sources for name in public_theorems(source)]
            if problem == 13:
                for name in ("IntegerListWork", "HeapRuntimeCostStrings", "HeapRuntimeCostComprehensions", "HeapRuntimeCostSumGenerators"):
                    names.extend(public_theorems(FORMALSW / f"lang/py/FormalSW/{name}.lean"))
            if problem == 18:
                for name in ("HeapRuntimeCostListCopies", "HeapRuntimeCostListLoops"):
                    names.extend(public_theorems(FORMALSW / f"lang/py/FormalSW/{name}.lean"))
            if problem == 19:
                for name in ("HeapRuntimeCostRangeSums", "HeapRuntimeCostRecords", "HeapRuntimeCostMembership"):
                    names.extend(public_theorems(FORMALSW / f"lang/py/FormalSW/{name}.lean"))
            if problem == 20:
                names.extend(public_theorems(FORMALSW / "lang/py/FormalSW/FactorialWorkBounds.lean"))
                names.extend(public_theorems(FORMALSW / "lang/py/FormalSW/HeapRuntimeCostLibrary.lean"))
            if problem in (7, 10):
                names.extend(public_theorems(ROOT / "data/project-euler-statements/data/lean/ProjectEulerStatements/Lib/ReciprocalSum.lean"))
            if not names:
                raise RuntimeError(f"No theorem declarations found for problem {problem}")
            imports = "".join(f'import {frontend["lean_module_for_path"](source)}\n' for source in sources)
            audited += audit_group(f"problem {problem}", imports, names)
        if 7 in problems:
            # The module interface avoids loading Bertrand's private analytic dependencies.
            sources = [ROOT / "data/project-euler-statements/data/lean/ProjectEulerStatements/Lib/NthPrimeBounds.lean",
                       ROOT / "ProjectEulerSolutions/Equiv/P7.lean"]
            names = [name for source in sources for name in public_theorems(source)]
            audited += audit_group("prime bounds", "module\nimport ProjectEulerSolutions.Equiv.P7\n", names)
    print(f"Audited {audited} theorems: only standard Lean axioms.")
    print("Complexity proofs and cost-model regressions passed.")


if __name__ == "__main__":
    main()
