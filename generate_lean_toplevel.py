#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
SOLVERS_DIR = ROOT / "solvers"
LIB_DIR = ROOT / "ProjectEulerSolutions"
OUTPUT = ROOT / "ProjectEulerSolutions.lean"

IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def module_name(stem: str) -> str:
    if IDENT_RE.match(stem):
        return stem
    return f"«{stem}»"


def sort_key(stem: str):
    if stem.isdigit():
        return (0, int(stem))
    if stem.startswith("P") and stem[1:].isdigit():
        return (0, int(stem[1:]))
    return (1, stem)


def main() -> int:
    lib_stems = [p.stem for p in LIB_DIR.glob("P*.lean")]
    solver_stems = [p.stem for p in SOLVERS_DIR.glob("*.lean")]
    solver_stems = [s for s in solver_stems if s != "ProjectEulerSolutions"]
    lib_stems.sort(key=sort_key)
    solver_stems.sort(key=sort_key)

    lines: list[str] = []
    lines.append("import Lean")
    for stem in lib_stems:
        lines.append(f"import ProjectEulerSolutions.{module_name(stem)}")

    lines.append("")
    lines.append("open Lean Elab Command Meta")
    lines.append("")
    lines.append(
        "private def isSolveApp (e : Expr) (solveName : Name) : Bool :="
    )
    lines.append("  let e := e.consumeMData")
    lines.append("  match e.getAppFn with")
    lines.append("  | Expr.const name _ => name == solveName")
    lines.append("  | _ => false")
    lines.append("")
    lines.append(
        "private def isSerializeSolve (e : Expr) (solveName : Name) : Bool :="
    )
    lines.append("  let e := e.consumeMData")
    lines.append("  match e.getAppFn with")
    lines.append("  | Expr.const name _ =>")
    lines.append("      let nameStr := name.toString")
    lines.append(
        "      if !nameStr.endsWith \".serialize\" && nameStr != \"serialize\" then"
    )
    lines.append("        false")
    lines.append("      else")
    lines.append("        let args := e.getAppArgs")
    lines.append("        match args[(args.size - 1)]? with")
    lines.append("        | some last => isSolveApp last solveName")
    lines.append("        | none => false")
    lines.append("  | _ => false")
    lines.append("")
    lines.append(
        "/- Check that `main` prints either `solve ...` or `serialize (solve ...)`"
    )
    lines.append(
        "   without caring about the number of arguments to `solve`. -/"
    )
    lines.append(
        "private def isPrintSolve (e : Expr) (solveName : Name) : Bool :="
    )
    lines.append("  let e := e.consumeMData")
    lines.append("  match e.getAppFn with")
    lines.append("  | Expr.const name _ =>")
    lines.append("      if name != ``IO.println then")
    lines.append("        false")
    lines.append("      else")
    lines.append("        let args := e.getAppArgs")
    lines.append("        match args[(args.size - 1)]? with")
    lines.append(
        "        | some last => isSolveApp last solveName || isSerializeSolve last solveName"
    )
    lines.append("        | none => false")
    lines.append("  | _ => false")
    lines.append("")
    lines.append(
        "syntax (name := checkMainPrintsSolve) \"check_main_prints_solve \" term \" with \" term : command"
    )
    lines.append("")
    lines.append("elab_rules : command")
    lines.append("  | `(check_main_prints_solve $mainTerm with $solveTerm) => do")
    lines.append("      liftTermElabM do")
    lines.append("        let mainExpr ← Term.elabTerm mainTerm none")
    lines.append("        let solveExpr ← Term.elabTerm solveTerm none")
    lines.append("        let mainName ←")
    lines.append("          match mainExpr with")
    lines.append("          | Expr.const name _ => pure name")
    lines.append("          | _ => throwError \"main must be a constant name\"")
    lines.append("        let solveName ←")
    lines.append("          match solveExpr with")
    lines.append("          | Expr.const name _ => pure name")
    lines.append("          | _ => throwError \"solve must be a constant name\"")
    lines.append("        let env ← getEnv")
    lines.append("        let some mainDecl := env.find? mainName")
    lines.append(
        "          | throwError m!\"unknown main definition '{mainName}'\""
    )
    lines.append("        let some mainVal := mainDecl.value?")
    lines.append("          | throwError m!\"main '{mainName}' has no value\"")
    lines.append("        let mainBody := mainVal")
    lines.append("        unless isPrintSolve mainBody solveName do")
    lines.append("          throwError")
    lines.append(
        "            m!\"main '{mainName}' is not of the form 'IO.println (solve ...)'\""
    )
    lines.append("")

    for stem in lib_stems:
        if not stem.startswith("P") or not stem[1:].isdigit():
            continue
        n = stem[1:]
        lines.append(f"def p{n}solve := ProjectEulerSolutions.P{n}.solve")
        lines.append("")
        lines.append(
            f"def p{n}equiv : ProjectEulerStatements.P{n}.naive = p{n}solve := by"
        )
        lines.append("  apply (funext_iff).2")
        lines.append(
            f"  simpa [funext_iff] using ProjectEulerSolutions.P{n}.equiv"
        )
        lines.append("")

    OUTPUT.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
