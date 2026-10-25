import solvers.«65.source»
import data.«project-euler-statements».data.py.project_euler_statements.«p65.py»

namespace solvers.«65».solve
open FormalSW.PyAst
set_option Elab.async false
private abbrev program := Solvers.P65Ast.module
private abbrev inputEnv (n : Int) : ScalarEnv := fun name => if name = "n" then some (.int n) else none

private theorem positive (n : Int)
    (condition : Calls.Condition program (inputEnv n) Solvers.P65Ast.requires_solve_0) : 1 ≤ n := by
  simpa [evalCmpOp] using Calls.Evaluates.condition_int_compare condition rfl

theorem «Returns» (n : Int) (condition : Calls.Condition program (inputEnv n) Solvers.P65Ast.requires_solve_0) :
    (HeapCalls.scalarFunction program "solve" [.int n]).Returns := by
  have valid := positive n condition
  lift n to Nat using (by omega : 0 ≤ n)
  exact (result n []).scalar_returns (values := [.int n])

theorem «Equiv(naive)» (n : Int)
    (condition : Calls.Condition program (inputEnv n) Solvers.P65Ast.requires_solve_0) :
    (HeapCalls.scalarFunction program "solve" [.int n]).Equiv
      (HeapCalls.scalarFunction P65Ast.module "naive" [.int n]) := by
  have valid := positive n condition
  lift n to Nat using (by omega : 0 ≤ n)
  obtain ⟨_, reference⟩ := data.«project-euler-statements».data.py.project_euler_statements.p65.naive.result n []
  have left := (result n []).scalar_finishes (values := [.int n])
  have right := reference.scalar_finishes (values := [.int n])
  intro outcome
  constructor
  · intro finished
    rw [finished.unique left]
    exact right
  · intro finished
    rw [finished.unique right]
    exact left

theorem «Supported» :
    (∀ n : Int, Calls.Condition program (inputEnv n) Solvers.P65Ast.requires_solve_0 →
      (HeapCalls.scalarFunction program "solve" [.int n]).Supported) ∧
    (∀ n : Int, Calls.Condition program (inputEnv n) Solvers.P65Ast.requires_solve_0 →
      (HeapCalls.scalarFunction P65Ast.module "naive" [.int n]).Supported) := by
  constructor
  · intro n condition
    exact («Returns» n condition).supported
  · intro n condition
    have valid := positive n condition
    lift n to Nat using (by omega : 0 ≤ n)
    obtain ⟨_, returned⟩ := data.«project-euler-statements».data.py.project_euler_statements.p65.naive.result n []
    exact (returned.scalar_returns (values := [.int n])).supported

theorem default_returns : (HeapCalls.scalarFunction program "solve" []).Returns := by
  exact (default_result []).scalar_returns (values := [])

end solvers.«65».solve
