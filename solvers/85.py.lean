import solvers.«85.py.lean.d».Execution
import data.«project-euler-statements».data.py.project_euler_statements.«p85.py»

/-! Termination and equivalence of the actual Python searches for every integer target. -/
namespace solvers.«85».solve
open FormalSW.PyAst ProjectEulerSolutions.RectangleSearch
set_option Elab.async false

theorem «Returns» (target : Int) :
    (HeapCalls.scalarFunction program "solve" [.int target]).Returns := by
  obtain ⟨_, _, _, returned, _⟩ := Optimized.solve_result target []
  exact returned.scalar_returns (values := [.int target])

theorem «Equiv(naive)» (target : Int) :
    (HeapCalls.scalarFunction program "solve" [.int target]).Equiv
      (Calls.function P85Ast.module "naive" [.int target]) := by
  obtain ⟨final, referenceRun, optimizedRun, _⟩ := searches_agree target
  have reference := data.«project-euler-statements».data.py.project_euler_statements.p85.Scalar.naive_result_of_run target final referenceRun
  obtain ⟨_, optimized, _⟩ := Optimized.solve_result_of_run target final optimizedRun []
  have left := optimized.scalar_finishes (values := [.int target])
  have right := reference.finishes
  intro outcome
  constructor
  · intro finished
    rw [finished.unique left]
    exact right
  · intro finished
    rw [finished.unique right]
    exact left

theorem «Supported» :
    (∀ target : Int, (HeapCalls.scalarFunction program "solve" [.int target]).Supported) ∧
    (∀ target : Int, (Calls.function P85Ast.module "naive" [.int target]).Supported) := by
  constructor
  · intro target
    exact («Returns» target).supported
  · exact data.«project-euler-statements».data.py.project_euler_statements.p85.naive.«Supported»

end solvers.«85».solve
