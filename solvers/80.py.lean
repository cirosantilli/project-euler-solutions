import solvers.«80.py.lean.d».Execution
import data.«project-euler-statements».data.py.project_euler_statements.«p80.py»

namespace solvers.«80».solve
open FormalSW.PyAst
set_option Elab.async false
private abbrev program := Solvers.P80Ast.module
private abbrev limitCondition (limit : Int) :=
  Calls.Condition program (fun name => if name = "limit" then some (.int limit) else none) Solvers.P80Ast.requires_solve_0
private abbrev digitsCondition (digits : Int) :=
  Calls.Condition program (fun name => if name = "digits" then some (.int digits) else none) Solvers.P80Ast.requires_solve_1

private theorem inputs (limit digits : Int) (bound : limitCondition limit) (precision : digitsCondition digits) :
    0 ≤ limit ∧ 0 ≤ digits := by
  have lower : 0 ≤ limit := by
    simpa [evalCmpOp] using Calls.Evaluates.condition_int_compare bound rfl
  have positive : 1 ≤ digits := by
    simpa [evalCmpOp] using Calls.Evaluates.condition_int_compare precision rfl
  omega

theorem «Returns» (limit digits : Int) (bound : limitCondition limit) (precision : digitsCondition digits) :
    (HeapCalls.scalarFunction program "solve" [.int limit, .int digits]).Returns := by
  obtain ⟨lower, positive⟩ := inputs limit digits bound precision
  lift limit to Nat using lower
  lift digits to Nat using positive
  exact (result limit digits []).scalar_returns (values := [.int limit, .int digits])

theorem «Equiv(naive)» (limit digits : Int) (bound : limitCondition limit) (precision : digitsCondition digits) :
    (HeapCalls.scalarFunction program "solve" [.int limit, .int digits]).Equiv
      (HeapCalls.scalarFunction P80Ast.module "naive" [.int limit, .int digits]) := by
  obtain ⟨lower, positive⟩ := inputs limit digits bound precision
  lift limit to Nat using lower
  lift digits to Nat using positive
  obtain ⟨_, reference⟩ := data.«project-euler-statements».data.py.project_euler_statements.p80.naive.result limit digits []
  have left := (result limit digits []).scalar_finishes (values := [.int limit, .int digits])
  have right := reference.scalar_finishes (values := [.int limit, .int digits])
  intro outcome
  constructor
  · intro finished
    rw [finished.unique left]
    exact right
  · intro finished
    rw [finished.unique right]
    exact left

theorem «Supported» :
    (∀ limit digits : Int, limitCondition limit → digitsCondition digits →
      (HeapCalls.scalarFunction program "solve" [.int limit, .int digits]).Supported) ∧
    (∀ limit digits : Int, limitCondition limit → digitsCondition digits →
      (HeapCalls.scalarFunction P80Ast.module "naive" [.int limit, .int digits]).Supported) := by
  constructor
  · intro limit digits bound precision
    exact («Returns» limit digits bound precision).supported
  · intro limit digits bound precision
    obtain ⟨lower, positive⟩ := inputs limit digits bound precision
    lift limit to Nat using lower
    lift digits to Nat using positive
    obtain ⟨_, returned⟩ := data.«project-euler-statements».data.py.project_euler_statements.p80.naive.result limit digits []
    exact (returned.scalar_returns (values := [.int limit, .int digits])).supported

theorem default_returns : (HeapCalls.scalarFunction program "solve" []).Returns :=
  (default_result []).scalar_returns (values := [])

end solvers.«80».solve
