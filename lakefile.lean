import Lake
open System Lake DSL

package ProjectEulerSolvers where
  version := v!"0.1.0"

lean_lib ProjectEulerSolvers

require ProjectEulerStatements from "data/project-euler-statements/data/lean"

def solverLeanInputs (pkg : Package) : FetchM (Job (Array FilePath)) := do
  inputDir (pkg.dir / "solvers") true (fun p => p.extension == some "lean")

def runMakeLeanSolvers (pkg : Package) : JobM PUnit := do
  let buildProc ← IO.Process.spawn {
    cmd := "lake"
    args := #[
      "-d",
      (pkg.dir / "data" / "project-euler-statements" / "data" / "lean").toString,
      "build",
      "ProjectEulerStatements:static"
    ]
    stdout := .inherit
    stderr := .inherit
  }
  let buildExitCode ← buildProc.wait
  if buildExitCode != 0 then
    Lake.error s!"lake build ProjectEulerStatements:static failed with exit code {buildExitCode}"

  let proc ← IO.Process.spawn {
    cmd := "lake"
    args := #[
      "env",
      "make",
      "-C",
      (pkg.dir / "solvers").toString,
      "lean"
    ]
    stdout := .inherit
    stderr := .inherit
  }
  let exitCode ← proc.wait
  if exitCode != 0 then
    Lake.error s!"lake env make failed with exit code {exitCode}"
  return PUnit.unit

@[default_target]
target leanSolvers pkg : FilePath := do
  let stamp := pkg.dir / "solvers" / ".lean-build-stamp"
  let deps ← solverLeanInputs pkg
  buildFileAfterDep stamp deps (fun _ => do
    let _ ← runMakeLeanSolvers pkg
    createParentDirs stamp
    IO.FS.writeFile stamp ""
    return PUnit.unit
  )
