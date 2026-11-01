import solvers.«63.py.lean.d».Execution
import data.«project-euler-statements».data.py.project_euler_statements.«p63.py»

set_option Elab.async false

namespace solvers.«63».solve
open FormalSW.PyAst
private abbrev program := Solvers.P63Ast.module
private abbrev inputEnv (base : Int) : ScalarEnv :=
  fun name => if name = "base" then some (.int base) else none

private theorem valid_base (base : Int)
    (condition : Calls.Condition program (inputEnv base) Solvers.P63Ast.requires_solve_0) : 2 ≤ base := by
  simpa [evalCmpOp] using Calls.Evaluates.condition_int_compare condition rfl

theorem «Returns» (base : Int)
    (condition : Calls.Condition program (inputEnv base) Solvers.P63Ast.requires_solve_0) :
    (Calls.function program "solve" [.int base]).Returns := by
  have valid := valid_base base condition
  lift base to Nat using (by omega : 0 ≤ base)
  exact (result base (by omega)).returns

theorem «Equiv(naive)» (base : Int)
    (condition : Calls.Condition program (inputEnv base) Solvers.P63Ast.requires_solve_0) :
    (Calls.function program "solve" [.int base]).Equiv
      (HeapCalls.scalarFunction P63Ast.module "naive" [.int base]) := by
  have valid := valid_base base condition
  lift base to Nat using (by omega : 0 ≤ base)
  obtain ⟨_, reference⟩ := data.«project-euler-statements».data.py.project_euler_statements.p63.naive.result
    base (by omega) []
  have left := (result base (by omega)).finishes
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
    (∀ base : Int, Calls.Condition program (inputEnv base) Solvers.P63Ast.requires_solve_0 →
      (Calls.function program "solve" [.int base]).Supported) ∧
    (∀ base : Int, Calls.Condition program (inputEnv base) Solvers.P63Ast.requires_solve_0 →
      (HeapCalls.scalarFunction P63Ast.module "naive" [.int base]).Supported) := by
  constructor
  · intro base condition
    exact («Returns» base condition).supported
  · intro base condition
    have valid := valid_base base condition
    lift base to Nat using (by omega : 0 ≤ base)
    obtain ⟨_, returned⟩ := data.«project-euler-statements».data.py.project_euler_statements.p63.naive.result
      base (by omega) []
    exact (returned.scalar_returns (values := [.int base])).supported

theorem default_returns : (Calls.function program "solve" []).Returns := default_result.returns

end solvers.«63».solve
