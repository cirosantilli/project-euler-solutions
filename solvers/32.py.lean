import solvers.«32.py.lean.d».Search
import data.«project-euler-statements».data.py.project_euler_statements.«p32.py»

set_option Elab.async false

namespace solvers.«32»
open FormalSW.PyAst ProjectEulerSolutions.PandigitalProductSearch

theorem solve.«Returns» (width : Int) :
    (SetCalls.call Solvers.P32Ast.module "solve" [.scalar (.int width)] []).Returns := by
  intro choices
  obtain ⟨heap, returned⟩ := solve.result_int width []
  exact ⟨.scalar (.int (answer width.toNat)), [], heap, 0, returned.finishes choices⟩

theorem solve.«Equiv(naive)» (width : Int) :
    (SetCalls.call Solvers.P32Ast.module "solve" [.scalar (.int width)] []).Equiv
      (SetCalls.call P32Ast.module "naive" [.scalar (.int width)] []) := by
  intro leftChoices rightChoices
  obtain ⟨optimizedHeap, optimized⟩ := solve.result_int width []
  obtain ⟨referenceHeap, reference⟩ :=
    data.«project-euler-statements».data.py.project_euler_statements.p32.naive.result_int width []
  apply Mutable.Call.equiv_of_finishes (optimized.finishes leftChoices) (reference.finishes rightChoices)
  exact Mutable.Completion.equivalent_allocations [] optimizedHeap referenceHeap
    (.scalar (.int (answer width.toNat))) [] (by simp [Mutable.Heap.Valid]) trivial (by simp)

theorem solve.«Supported» :
    (∀ width : Int, (SetCalls.call Solvers.P32Ast.module "solve" [.scalar (.int width)] []).Supported) ∧
    (∀ width : Int, (SetCalls.call P32Ast.module "naive" [.scalar (.int width)] []).Supported) := by
  constructor
  · intro width choices
    exact (solve.«Returns» width choices).supported
  · exact data.«project-euler-statements».data.py.project_euler_statements.p32.naive.«Supported»

end solvers.«32»
