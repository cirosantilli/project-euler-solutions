import solvers.«88.py.lean.d».Execution
import data.«project-euler-statements».data.py.project_euler_statements.«p88.py»

namespace solvers.«88».solve
open FormalSW.PyAst ProjectEulerSolutions.ProductSumNumbers
set_option Elab.async false

theorem «Returns» (maxSize : Int) :
    (SetCalls.call program "solve" [integer maxSize] []).Returns := by
  intro choices
  obtain ⟨heap, returned⟩ := result maxSize []
  exact ⟨number (answer maxSize.toNat), [], heap, 0, returned.finishes choices⟩

/-- Both complete Python programs return the same answer for every signed bound. -/
theorem «Equiv(naive)» (maxSize : Int) :
    (SetCalls.call program "solve" [integer maxSize] []).Equiv
      (SetCalls.call P88Ast.module "naive" [integer maxSize] []) := by
  intro leftChoices rightChoices
  obtain ⟨leftHeap, optimized⟩ := result maxSize []
  obtain ⟨rightHeap, reference⟩ :=
    data.«project-euler-statements».data.py.project_euler_statements.p88.naive.result maxSize []
  apply Mutable.Call.equiv_of_finishes
    (a := (SetCalls.call program "solve" [integer maxSize] []).run leftChoices)
    (b := (SetCalls.call P88Ast.module "naive" [integer maxSize] []).run rightChoices)
    (optimized.finishes leftChoices)
    (SetCalls.finishes_of_heap reference.finishes rightChoices 0)
  exact Mutable.Completion.equivalent_allocations [] leftHeap rightHeap
    (number (answer maxSize.toNat)) [] (by simp [Mutable.Heap.Valid]) trivial (by simp)

theorem «Supported» :
    (∀ maxSize : Int, (SetCalls.call program "solve" [integer maxSize] []).Supported) ∧
    (∀ maxSize : Int, (SetCalls.call P88Ast.module "naive" [integer maxSize] []).Supported) := by
  constructor
  · intro maxSize choices
    exact («Returns» maxSize choices).supported
  · intro maxSize choices
    obtain ⟨_, returned⟩ :=
      data.«project-euler-statements».data.py.project_euler_statements.p88.naive.result maxSize []
    exact (SetCalls.returns_of_heap returned.returns choices).supported

end solvers.«88».solve
