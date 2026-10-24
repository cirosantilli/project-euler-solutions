import solvers.«38.search.source»
import data.«project-euler-statements».data.py.project_euler_statements.«p38.py»

namespace solvers.«38».solve
open FormalSW.PyAst
private abbrev program := Solvers.P38Ast.module

theorem «Returns» (width : Int) : (HeapCalls.scalarFunction program "solve" [.int width]).Returns := by
  by_cases negative : width < 0
  · exact (empty (globals := []) width (by omega) []).scalar_returns (values := [.int width])
  · lift width to Nat using (by omega : 0 ≤ width)
    obtain ⟨_, returned⟩ := result (globals := []) width []
    exact returned.scalar_returns (values := [.int width])

theorem «Equiv(naive)» (width : Int) :
    (HeapCalls.scalarFunction program "solve" [.int width]).Equiv
      (HeapCalls.scalarFunction P38Ast.module "naive" [.int width]) := by
  by_cases negative : width < 0
  · have optimized := (empty (globals := []) width (by omega) []).scalar_finishes (values := [.int width])
    have reference := (data.«project-euler-statements».data.py.project_euler_statements.p38.naive.empty
      (globals := []) width (by omega) []).scalar_finishes (values := [.int width])
    intro outcome
    constructor
    · intro finished
      rw [finished.unique optimized]
      exact reference
    · intro finished
      rw [finished.unique reference]
      exact optimized
  · lift width to Nat using (by omega : 0 ≤ width)
    obtain ⟨_, optimizedRun⟩ := result (globals := []) width []
    obtain ⟨_, referenceRun⟩ := data.«project-euler-statements».data.py.project_euler_statements.p38.naive.result (globals := []) width []
    have optimized := optimizedRun.scalar_finishes (values := [.int width])
    have reference := referenceRun.scalar_finishes (values := [.int width])
    intro outcome
    constructor
    · intro finished
      rw [finished.unique optimized]
      exact reference
    · intro finished
      rw [finished.unique reference]
      exact optimized

theorem «Supported» :
    (∀ width : Int, (HeapCalls.scalarFunction program "solve" [.int width]).Supported) ∧
    (∀ width : Int, (HeapCalls.scalarFunction P38Ast.module "naive" [.int width]).Supported) := by
  constructor
  · intro width
    exact («Returns» width).supported
  · intro width
    exact (data.«project-euler-statements».data.py.project_euler_statements.p38.naive.«Returns» width).supported

theorem default_returns : (HeapCalls.scalarFunction program "solve" []).Returns := by
  obtain ⟨finalHeap, returned⟩ := result (globals := []) 9 []
  have defaultRun : HeapCalls.ReturnsValue program "solve" [] []
      (.scalar (.int (ProjectEulerSolutions.PandigitalConcat.reference 9))) finalHeap :=
    HeapCalls.ReturnsValue.with_defaults rfl rfl rfl returned
  exact defaultRun.scalar_returns (values := [])

end solvers.«38».solve
