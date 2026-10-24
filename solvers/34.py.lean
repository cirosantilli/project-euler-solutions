import solvers.«34.search.source»

namespace solvers.«34».solve
open FormalSW.PyAst
private abbrev inputEnv (base : Int) : ScalarEnv := fun name => if name = "base" then some (.int base) else none
private abbrev program := Solvers.P34Ast.module

private theorem valid_base (base : Int) (condition : Calls.Condition program (inputEnv base) Solvers.P34Ast.requires_solve_0) : 2 ≤ base := by
  simpa [evalCmpOp] using Calls.Evaluates.condition_int_compare condition rfl

theorem «Returns» (base : Int) (condition : Calls.Condition program (inputEnv base) Solvers.P34Ast.requires_solve_0) :
    (HeapCalls.scalarFunction program "solve" [.int base]).Returns := by
  have valid := valid_base base condition
  lift base to Nat using (by omega : 0 ≤ base)
  obtain ⟨_, returned⟩ := result base (by omega) []
  exact returned.scalar_returns (values := [.int base])

theorem «Equiv(naive)» (base : Int) (condition : Calls.Condition program (inputEnv base) Solvers.P34Ast.requires_solve_0) :
    (HeapCalls.scalarFunction program "solve" [.int base]).Equiv
      (HeapCalls.scalarFunction P34Ast.module "naive" [.int base]) := by
  have valid := valid_base base condition
  lift base to Nat using (by omega : 0 ≤ base)
  obtain ⟨_, optimized⟩ := result base (by omega) []
  obtain ⟨_, reference⟩ := data.«project-euler-statements».data.py.project_euler_statements.p34.naive.result base (by omega) []
  have left := optimized.scalar_finishes (values := [.int base])
  have right := reference.scalar_finishes (values := [.int base])
  intro outcome
  constructor
  · intro finished
    rw [finished.unique left]
    exact right
  · intro finished
    rw [finished.unique right]
    exact left

theorem «Supported» :
    (∀ base : Int, Calls.Condition program (inputEnv base) Solvers.P34Ast.requires_solve_0 →
      (HeapCalls.scalarFunction program "solve" [.int base]).Supported) ∧
    (∀ base : Int, Calls.Condition program (inputEnv base) Solvers.P34Ast.requires_solve_0 →
      (HeapCalls.scalarFunction P34Ast.module "naive" [.int base]).Supported) := by
  constructor
  · intro base condition
    exact («Returns» base condition).supported
  · intro base condition
    have valid := valid_base base condition
    lift base to Nat using (by omega : 0 ≤ base)
    obtain ⟨_, returned⟩ := data.«project-euler-statements».data.py.project_euler_statements.p34.naive.result base (by omega) []
    exact (returned.scalar_returns (values := [.int base])).supported

theorem default_returns : (HeapCalls.scalarFunction program "solve" []).Returns := by
  obtain ⟨finalHeap, returned⟩ := result 10 (by decide) []
  have defaultRun : HeapCalls.ReturnsValue program "solve" [] []
      (.scalar (.int (ProjectEulerSolutions.DigitFactorialBase.answer 10 (by decide)))) finalHeap :=
    HeapCalls.ReturnsValue.with_defaults rfl rfl rfl returned
  exact defaultRun.scalar_returns (values := [])

end solvers.«34».solve
