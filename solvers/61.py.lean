import solvers.«61.py.lean.d».Main
import data.«project-euler-statements».data.py.project_euler_statements.«p61.py»

set_option Elab.async false

namespace solvers.«61».solve
open FormalSW.PyAst ProjectEulerSolutions

theorem «Equiv(naive)» (maxSides : Int) :
    (HeapCalls.scalarFunction Solvers.P61Ast.module "solve" [.int maxSides]).Equiv
      (HeapCalls.scalarFunction P61Ast.module "naive" [.int maxSides]) := by
  obtain ⟨_, optimized⟩ := heap_result maxSides []
  obtain ⟨_, reference⟩ :=
    data.«project-euler-statements».data.py.project_euler_statements.p61.naive.heap_result maxSides []
  have left := optimized.scalar_finishes (values := [.int maxSides])
  have right := reference.scalar_finishes (values := [.int maxSides])
  have same := PolygonalCycleSearch.models_equal PolygonalFamilies.family
    (PolygonalCycleSearch.indexed PolygonalFamilies.family) (fun _ _ => rfl) maxSides
  simp only [answer, same] at left
  intro outcome
  constructor
  · intro finished
    rw [finished.unique left]
    exact right
  · intro finished
    rw [finished.unique right]
    exact left

theorem «Supported» :
    (∀ maxSides : Int, (HeapCalls.scalarFunction Solvers.P61Ast.module "solve" [.int maxSides]).Supported) ∧
    (∀ maxSides : Int, (HeapCalls.scalarFunction P61Ast.module "naive" [.int maxSides]).Supported) :=
  ⟨fun maxSides => («Returns» maxSides).supported,
    data.«project-euler-statements».data.py.project_euler_statements.p61.naive.«Supported»⟩

end solvers.«61».solve
