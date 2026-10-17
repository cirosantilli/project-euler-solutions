import «.formalsw».solvers.«6.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p6.py»
import FormalSW
import ProjectEulerSolutions.Equiv.P6

namespace solvers.«6»
open FormalSW.PyAst FormalSW.PyAst.Calls
open scoped BigOperators
private abbrev program := Solvers.P6Ast.module

private theorem squares_le (n : Nat) :
    ProjectEulerSolutions.P6.sumOfSquares n ≤ ProjectEulerSolutions.P6.squareOfSum n := by
  have h := data.«project-euler-statements».data.py.project_euler_statements.p6.sum_squares_le n
  change (∑ i ∈ Finset.Icc 1 n, i ^ 2) ≤ (∑ i ∈ Finset.Icc 1 n, i) ^ 2 at h
  rw [ProjectEulerSolutions.P6.sum_Icc_sq, ProjectEulerSolutions.P6.sum_Icc_id] at h
  simpa [ProjectEulerSolutions.P6.sumOfSquares, ProjectEulerSolutions.P6.squareOfSum, pow_two] using h

theorem solve.result (n : Nat) :
    Finishes (scalarFunction program "solve" [.int n]).step
      (scalarFunction program "solve" [.int n]).initial
      (.ok (.int (ProjectEulerSolutions.P6.solve n))) := by
  apply Execution.finishes_of_observe (fuel := 3)
  change Execution.Observation.done (.ok (Constant.int (((n : Int) * (n + 1) / 2) * (n * (n + 1) / 2) -
    n * (n + 1) * (2 * n + 1) / 6))) = _
  simp only [ProjectEulerSolutions.P6.solve, Int.natCast_sub (squares_le n)]
  simp [ProjectEulerSolutions.P6.squareOfSum, ProjectEulerSolutions.P6.sumOfSquares]

theorem solve.«Equiv(naive)» (n : Int) (hp : Condition program
    (fun key => if key = "n" then some (.int n) else none) Solvers.P6Ast.requires_solve_0) :
    (scalarFunction program "solve" [.int n]).Equiv (function P6Ast.module "naive" [.int n]) := by
  have nonnegative : 0 ≤ n := by
    simpa [evalCmpOp] using Evaluates.condition_int_compare hp rfl
  lift n to Nat using nonnegative
  have reference := data.«project-euler-statements».data.py.project_euler_statements.p6.naive.result n
  rw [ProjectEulerSolutions.P6.equiv] at reference
  have referenceRun := reference.finishes
  have optimizedRun := solve.result n
  intro outcome
  constructor
  · intro run
    rw [run.unique optimizedRun]
    exact referenceRun
  · intro run
    rw [run.unique referenceRun]
    exact optimizedRun

theorem solve.«Supported» :
    (∀ (n : Int), Condition program (fun key => if key = "n" then some (.int n) else none)
      Solvers.P6Ast.requires_solve_0 → (scalarFunction program "solve" [.int n]).Supported) ∧
    (∀ (n : Int), Condition program (fun key => if key = "n" then some (.int n) else none)
      Solvers.P6Ast.requires_solve_0 → (function P6Ast.module "naive" [.int n]).Supported) := by
  constructor
  · intro n hp
    have nonnegative : 0 ≤ n := by
      simpa [evalCmpOp] using Evaluates.condition_int_compare hp rfl
    lift n to Nat using nonnegative
    exact (show (scalarFunction program "solve" [.int n]).Returns from ⟨_, solve.result n⟩).supported
  · intro n hp
    have nonnegative : 0 ≤ n := by
      simpa [evalCmpOp] using Evaluates.condition_int_compare hp rfl
    lift n to Nat using nonnegative
    exact (data.«project-euler-statements».data.py.project_euler_statements.p6.naive.result n).returns.supported

end solvers.«6»
