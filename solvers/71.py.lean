import solvers.«71.py.lean.d».Execution
import data.«project-euler-statements».data.py.project_euler_statements.«p71.py»

/-! Universal Python source termination and equivalence for every proper target fraction. -/
namespace solvers.«71».solve
open FormalSW.PyAst
set_option Elab.async false

abbrev reference := P71Ast.module
open data.«project-euler-statements».data.py.project_euler_statements.p71

theorem «Returns» (limit num den : Int)
    (positive : positiveCondition program num) (proper : properCondition program num den) :
    (HeapCalls.scalarFunction program "solve" [.int limit, .int num, .int den]).Returns := by
  have positive := positive_of_condition program num positive
  have proper := proper_of_condition program num den proper
  obtain ⟨_, returned⟩ := result limit num den [] positive proper
  exact returned.scalar_returns (values := [.int limit, .int num, .int den])

theorem «Equiv(naive)» (limit num den : Int)
    (positive : positiveCondition program num) (proper : properCondition program num den) :
    (HeapCalls.scalarFunction program "solve" [.int limit, .int num, .int den]).Equiv
      (Calls.function reference "naive" [.int limit, .int num, .int den]) := by
  have positive := positive_of_condition program num positive
  have proper := proper_of_condition program num den proper
  obtain ⟨_, returned⟩ := result limit num den [] positive proper
  have left := returned.scalar_finishes (values := [.int limit, .int num, .int den])
  have right :=
    (data.«project-euler-statements».data.py.project_euler_statements.p71.naive.result
      limit num den positive proper).finishes
  rw [ProjectEulerSolutions.P71.solve_eq_reference limit num den positive proper] at left
  intro outcome
  constructor
  · intro finished
    rw [finished.unique left]
    exact right
  · intro finished
    rw [finished.unique right]
    exact left

theorem «Supported» :
    (∀ limit num den : Int, positiveCondition program num → properCondition program num den →
      (HeapCalls.scalarFunction program "solve" [.int limit, .int num, .int den]).Supported) ∧
    (∀ limit num den : Int, positiveCondition program num → properCondition program num den →
      (Calls.function reference "naive" [.int limit, .int num, .int den]).Supported) := by
  constructor
  · intro limit num den positive proper
    exact («Returns» limit num den positive proper).supported
  · intro limit num den positive proper
    exact (data.«project-euler-statements».data.py.project_euler_statements.p71.naive.result
      limit num den (positive_of_condition program num positive)
      (proper_of_condition program num den proper)).returns.supported

theorem one_argument_returns (limit : Int) :
    (HeapCalls.scalarFunction program "solve" [.int limit]).Returns := by
  obtain ⟨_, returned⟩ := one_argument_result limit []
  exact returned.scalar_returns (values := [.int limit])

theorem one_argument_equiv (limit : Int) :
    (HeapCalls.scalarFunction program "solve" [.int limit]).Equiv
      (Calls.function reference "naive" [.int limit]) := by
  obtain ⟨_, returned⟩ := one_argument_result limit []
  have left := returned.scalar_finishes (values := [.int limit])
  have right :=
    (data.«project-euler-statements».data.py.project_euler_statements.p71.naive.one_argument_result
      limit).finishes
  rw [ProjectEulerSolutions.P71.solve_eq_reference limit 3 7 (by decide) (by decide)] at left
  intro outcome
  constructor
  · intro finished
    rw [finished.unique left]
    exact right
  · intro finished
    rw [finished.unique right]
    exact left

theorem two_argument_returns (limit num : Int) (positive : 0 < num) (proper : num < 7) :
    (HeapCalls.scalarFunction program "solve" [.int limit, .int num]).Returns := by
  obtain ⟨_, returned⟩ := two_argument_result limit num [] positive proper
  exact returned.scalar_returns (values := [.int limit, .int num])

theorem two_argument_equiv (limit num : Int) (positive : 0 < num) (proper : num < 7) :
    (HeapCalls.scalarFunction program "solve" [.int limit, .int num]).Equiv
      (Calls.function reference "naive" [.int limit, .int num]) := by
  obtain ⟨_, returned⟩ := two_argument_result limit num [] positive proper
  have left := returned.scalar_finishes (values := [.int limit, .int num])
  have right :=
    (data.«project-euler-statements».data.py.project_euler_statements.p71.naive.two_argument_result
      limit num positive proper).finishes
  rw [ProjectEulerSolutions.P71.solve_eq_reference limit num 7 positive proper] at left
  intro outcome
  constructor
  · intro finished
    rw [finished.unique left]
    exact right
  · intro finished
    rw [finished.unique right]
    exact left

end solvers.«71».solve
