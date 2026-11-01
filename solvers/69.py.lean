import solvers.«69.py.lean.d».Execution
import data.«project-euler-statements».data.py.project_euler_statements.«p69.py»

namespace solvers.«69».solve
open FormalSW.PyAst ProjectEulerStatements.TotientMaximum
set_option Elab.async false

theorem «Returns» (limit : Int) :
    (HeapCalls.scalarFunction program "solve" [.int limit]).Returns := by
  obtain ⟨_, returned⟩ := result limit []
  exact returned.scalar_returns (values := [.int limit])

theorem «Equiv(naive)» (limit : Int) :
    (HeapCalls.scalarFunction program "solve" [.int limit]).Equiv
      (Calls.function P69Ast.module "naive" [.int limit]) := by
  obtain ⟨_, returned⟩ := result limit []
  have left := returned.scalar_finishes (values := [.int limit])
  have right :=
    (data.«project-euler-statements».data.py.project_euler_statements.p69.naive.result
      limit).finishes
  rw [optimizedResult_eq] at left
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
    (∀ limit : Int, (Calls.function P69Ast.module "naive" [.int limit]).Supported) := by
  constructor
  · intro limit
    exact («Returns» limit).supported
  · intro limit
    exact (data.«project-euler-statements».data.py.project_euler_statements.p69.naive.«Returns»
      limit).supported

end solvers.«69».solve
