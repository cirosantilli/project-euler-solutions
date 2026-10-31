import solvers.«75.source»
import data.«project-euler-statements».data.py.project_euler_statements.«p75.py»

/-! Source termination and equivalence for every signed perimeter bound. -/
namespace solvers.«75».solve
open FormalSW.PyAst
set_option Elab.async false

theorem «Returns» (limit : Int) :
    (HeapCalls.scalarFunction program "solve" [.int limit]).Returns := by
  obtain ⟨_, returned⟩ := result limit []
  exact returned.scalar_returns (values := [.int limit])

theorem «Equiv(naive)» (limit : Int) :
    (HeapCalls.scalarFunction program "solve" [.int limit]).Equiv
      (HeapCalls.scalarFunction P75Ast.module "naive" [.int limit]) := by
  obtain ⟨_, returned⟩ := result limit []
  have optimized := returned.scalar_finishes (values := [.int limit])
  have reference :=
    (data.«project-euler-statements».data.py.project_euler_statements.p75.naive.result
      limit []).scalar_finishes (values := [.int limit])
  intro outcome
  constructor
  · intro finished
    rw [finished.unique optimized]
    exact reference
  · intro finished
    rw [finished.unique reference]
    exact optimized

theorem «Supported» :
    (∀ limit : Int, (HeapCalls.scalarFunction program "solve" [.int limit]).Supported) ∧
    (∀ limit : Int, (HeapCalls.scalarFunction P75Ast.module "naive" [.int limit]).Supported) := by
  constructor
  · intro limit
    exact («Returns» limit).supported
  · intro limit
    exact (data.«project-euler-statements».data.py.project_euler_statements.p75.naive.«Returns»
      limit).supported

end solvers.«75».solve
