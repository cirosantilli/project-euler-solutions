import solvers.«68.source»
import data.«project-euler-statements».data.py.project_euler_statements.«p68.py»
import lang.py.FormalSW.SetSumCalls

namespace solvers.«68».solve
open FormalSW.PyAst ProjectEulerStatements.MagicRing ProjectEulerStatements.MagicRingTextCalls
set_option Elab.async false

private theorem equivalent_results (values : List Mutable.Value) (n digits : Int)
    (left : ∃ finalHeap, HeapCalls.ReturnsValue program "solve" values []
      (string (referenceResult n digits)) finalHeap)
    (right : ∃ finalHeap, HeapCalls.ReturnsValue P68Ast.module "naive" values []
      (string (referenceResult n digits)) finalHeap) :
    (SetCalls.call program "solve" values []).Equiv
      (SetCalls.call P68Ast.module "naive" values []) := by
  intro leftChoices rightChoices
  obtain ⟨leftHeap, optimized⟩ := left
  obtain ⟨rightHeap, reference⟩ := right
  apply Mutable.Call.equiv_of_finishes
    ((SetCalls.ReturnsValue.of_heap optimized).finishes leftChoices)
    ((SetCalls.ReturnsValue.of_heap reference).finishes rightChoices)
  exact Mutable.Completion.equivalent_allocations [] leftHeap rightHeap
    (string (referenceResult n digits)) [] (by simp [Mutable.Heap.Valid]) trivial (by simp)

theorem «Returns» (n digits : Int) :
    (SetCalls.call program "solve" [integer n, integer digits] []).Returns := by
  obtain ⟨_, returned⟩ := full_result n digits []
  exact SetCalls.returns_of_heap returned.returns

theorem «Equiv(naive)» (n digits : Int) :
    (SetCalls.call program "solve" [integer n, integer digits] []).Equiv
      (SetCalls.call P68Ast.module "naive" [integer n, integer digits] []) :=
  equivalent_results _ n digits (full_result n digits [])
    (data.«project-euler-statements».data.py.project_euler_statements.p68.naive.full_result
      n digits [])

theorem «Supported» :
    (∀ n digits : Int, (SetCalls.call program "solve" [integer n, integer digits] []).Supported) ∧
    (∀ n digits : Int,
      (SetCalls.call P68Ast.module "naive" [integer n, integer digits] []).Supported) := by
  constructor
  · intro n digits choices
    obtain ⟨_, returned⟩ := full_result n digits []
    exact Execution.supported_of_finishes
      ((SetCalls.ReturnsValue.of_heap returned).finishes choices)
  · intro n digits choices
    obtain ⟨_, returned⟩ :=
      data.«project-euler-statements».data.py.project_euler_statements.p68.naive.full_result
        n digits []
    exact Execution.supported_of_finishes
      ((SetCalls.ReturnsValue.of_heap returned).finishes choices)

theorem one_argument_returns (n : Int) :
    (SetCalls.call program "solve" [integer n] []).Returns := by
  obtain ⟨_, returned⟩ := one_argument_result n []
  exact SetCalls.returns_of_heap returned.returns

theorem one_argument_equiv (n : Int) :
    (SetCalls.call program "solve" [integer n] []).Equiv
      (SetCalls.call P68Ast.module "naive" [integer n] []) :=
  equivalent_results _ n 16 (one_argument_result n [])
    (data.«project-euler-statements».data.py.project_euler_statements.p68.naive.one_argument_result
      n [])

theorem default_returns : (SetCalls.call program "solve" [] []).Returns := by
  obtain ⟨_, returned⟩ := default_result []
  exact SetCalls.returns_of_heap returned.returns

theorem default_equiv : (SetCalls.call program "solve" [] []).Equiv
    (SetCalls.call P68Ast.module "naive" [] []) :=
  equivalent_results _ 5 16 (default_result [])
    (data.«project-euler-statements».data.py.project_euler_statements.p68.naive.default_result [])

end solvers.«68».solve
