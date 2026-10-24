import solvers.«72.source»
import data.«project-euler-statements».data.py.project_euler_statements.«p72.py»

/-! Termination and equivalence of the totient-sieve and exhaustive Python solvers. -/
namespace solvers.«72»
open FormalSW.PyAst
set_option Elab.async false

theorem totients_up_to.«Returns» (limit : Int) :
    (HeapCalls.call program "totients_up_to" [.scalar (.int limit)] []).Returns := by
  obtain ⟨_, returned, _⟩ := totients_up_to.result limit []
  exact returned.returns

theorem totients_up_to.«Supported» (limit : Int) :
    (HeapCalls.call program "totients_up_to" [.scalar (.int limit)] []).Supported :=
  (totients_up_to.«Returns» limit).supported

namespace solve

theorem «Returns» (limit : Int) :
    (HeapCalls.scalarFunction program "solve" [.int limit]).Returns := by
  obtain ⟨_, returned⟩ := result limit []
  exact returned.scalar_returns (values := [.int limit])

theorem «Equiv(naive)» (limit : Int) :
    (HeapCalls.scalarFunction program "solve" [.int limit]).Equiv
      (Calls.function P72Ast.module "naive" [.int limit]) := by
  obtain ⟨_, returned⟩ := result limit []
  have left := returned.scalar_finishes (values := [.int limit])
  have right :=
    (data.«project-euler-statements».data.py.project_euler_statements.p72.naive.result
      limit).finishes
  intro outcome
  constructor
  · intro finished
    rw [finished.unique left]
    exact right
  · intro finished
    rw [finished.unique right]
    exact left

theorem «Supported» :
    (∀ limit : Int, (HeapCalls.scalarFunction program "solve" [.int limit]).Supported) ∧
    (∀ limit : Int, (Calls.function P72Ast.module "naive" [.int limit]).Supported) := by
  constructor
  · intro limit
    exact («Returns» limit).supported
  · intro limit
    exact (data.«project-euler-statements».data.py.project_euler_statements.p72.naive.«Returns»
      limit).supported

end solve
end solvers.«72»
