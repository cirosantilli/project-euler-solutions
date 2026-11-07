#!/usr/bin/env python3
"""
Bounded observations of the actual P1--P10 sources; not complexity proofs.

Run from the repository root: python3 -I -B tools/complexity_probe.py
Only standard-library dependencies. No Lean, Nagini, default solver imports,
bytecode caches, subprocesses, or source-file writes. Output is JSON.
"""

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import signal
import sys
import types


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data/project-euler-statements/formalsw"))
from lang.py.nagini_erasure import erase_nagini_annotations

REFERENCE = ROOT / "data/project-euler-statements/data/py/project_euler_statements"
PACKAGE = "_complexity_probe_sources"
EVENT_LIMIT = 2_000_000
SECONDS_LIMIT = 20


class Meter:
    def __init__(self):
        self.active = False
        self.counts = Counter()
        self.calls = Counter()
        self.files = set()

    def observe(self, value, kind):
        # Identity wrapper: the original source expression has already executed.
        if self.active:
            if kind == "slice" and isinstance(value, (list, str, bytes, bytearray)):
                self.counts["slice_items"] += len(value)
            elif kind == "add" and isinstance(value, list):
                self.counts["concat_items"] += len(value)
            elif kind == "slice_rhs" and isinstance(value, (bytes, bytearray)):
                self.counts["byte_slice_items"] += len(value)
        return value

    def trace(self, frame, event, arg):
        if frame.f_code.co_filename not in self.files:
            return None
        if event == "call":
            self.calls[frame.f_code.co_name] += 1
        elif event == "line":
            self.counts["lines"] += 1
            if self.counts["lines"] > EVENT_LIMIT:
                raise RuntimeError("probe source-line budget exhausted")
        return self.trace


class Instrument(ast.NodeTransformer):
    def wrap(self, node, kind):
        return ast.copy_location(
            ast.Call(
                func=ast.Name(id="_probe_observe", ctx=ast.Load()),
                args=[node, ast.Constant(kind)],
                keywords=[],
            ),
            node,
        )

    def visit_Subscript(self, node):
        self.generic_visit(node)
        if isinstance(node.ctx, ast.Load) and isinstance(node.slice, ast.Slice):
            return self.wrap(node, "slice")
        return node

    def visit_BinOp(self, node):
        self.generic_visit(node)
        return self.wrap(node, "add") if isinstance(node.op, ast.Add) else node

    def visit_Assign(self, node):
        self.generic_visit(node)
        if any(
            isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Slice)
            for target in node.targets
        ):
            node.value = self.wrap(node.value, "slice_rhs")
        return node


def load_source(path, name, meter, hashes, instrument=True):
    source = path.read_bytes()
    hashes[str(path.relative_to(ROOT))] = hashlib.sha256(source).hexdigest()
    tree = ast.parse(source, filename=str(path))
    tree.formalsw_source_text = source.decode("utf-8")
    erase_nagini_annotations(tree)
    # Fail closed if a later source change would try to import a proof dependency.
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            allowed = node.module in {"__future__", "math", "functools", "itertools"}
            allowed = allowed or (node.level == 1 and node.module == "common")
            if not allowed:
                raise RuntimeError(f"unreviewed import in {path}: {ast.unparse(node)}")
        elif isinstance(node, ast.Import):
            if any(alias.name != "math" for alias in node.names):
                raise RuntimeError(f"unreviewed import in {path}: {ast.unparse(node)}")
    if instrument:
        tree = Instrument().visit(tree)
    ast.fix_missing_locations(tree)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = PACKAGE
    module.__dict__["_probe_observe"] = meter.observe
    sys.modules[name] = module
    meter.files.add(str(path))
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


def timeout(signum, frame):
    raise RuntimeError("probe wall-clock budget exhausted")


def measure(function, args, meter):
    meter.counts.clear()
    meter.calls.clear()
    previous_trace = sys.gettrace()
    meter.active = True
    sys.settrace(meter.trace)
    try:
        try:
            result = {"value": function(*args)}
        except ValueError as error:
            result = {"exception": type(error).__name__, "message": str(error)}
    finally:
        sys.settrace(previous_trace)
        meter.active = False
    return {
        **result,
        **{key: meter.counts[key] for key in (
            "lines", "slice_items", "concat_items", "byte_slice_items"
        )},
        "calls": dict(sorted(meter.calls.items())),
    }


def cases():
    for problem, values in {
        1: [16, 32, 64, 128],
        2: [16, 256, 4096, 65536],
        3: [97, 101, 64, 128, 256, 1024],
        4: [0, 1, 2],
        5: [8, 16, 32, 64],
        6: [16, 32, 64, 128],
        7: [1, 6, 16, 32, 64],
        9: [12, 24, 60, 120, 11],
        10: [3, 5, 7, 9, 16, 32, 64, 128],
    }.items():
        for value in values:
            yield problem, str(value), (value,)
    for length in [16, 32, 64, 128]:
        yield 8, f"N={length}, k=4, all 2", ([2] * length, 4)
    for label, args in [
        ("N=64, k=32, all 2", ([2] * 64, 32)),
        ("N=64, k=65, all 2", ([2] * 64, 65)),
        ("N=64, k=4, zeros", ([0] * 64, 4)),
        ("N=64, k=4, alternating 0/2", ([0, 2] * 32, 4)),
        ("N=16, k=0, all 2", ([2] * 16, 0)),
        ("N=0, k=0", ([], 0)),
    ]:
        yield 8, label, args


def main():
    # One process, small hard-coded inputs, unchanged recursion limit, no workers.
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(SECONDS_LIMIT)
    meter = Meter()
    hashes = {}
    package = types.ModuleType(PACKAGE)
    package.__path__ = []  # Never search the filesystem for default-name imports.
    sys.modules[PACKAGE] = package
    load_source(REFERENCE / "common.py", f"{PACKAGE}.common", meter, hashes)
    functions = {}
    for problem in range(1, 11):
        reference = load_source(
            REFERENCE / f"p{problem}.py", f"{PACKAGE}.p{problem}", meter, hashes
        )
        solver = load_source(
            ROOT / "solvers" / f"{problem}.py", f"{PACKAGE}.solver{problem}", meter, hashes
        )
        functions[problem] = (reference.naive, solver.solve)
    rows = []
    for problem, label, args in cases():
        reference, solver = functions[problem]
        ref_result = measure(reference, args, meter)
        opt_result = measure(solver, args, meter)
        rows.append({
            "problem": problem,
            "input": label,
            "reference": ref_result,
            "solver": opt_result,
            "same_outcome": ref_result.get("value") == opt_result.get("value")
            and ref_result.get("exception") == opt_result.get("exception"),
        })
    signal.alarm(0)
    print(json.dumps({
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "line_budget_per_call": EVENT_LIMIT,
        "wall_seconds_budget": SECONDS_LIMIT,
        "p1_contracts_erased": True,
        "source_sha256": hashes,
        "rows": rows,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
