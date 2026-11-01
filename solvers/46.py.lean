import solvers.«46.py.lean.d».Search
import ProjectEulerSolutions.Lib.PrimeTwiceSquareSearch

namespace solvers.«46».solve
open FormalSW.PyAst
open ProjectEulerSolutions.PrimeTwiceSquareSuccessor

theorem «Equiv(naive)» (after : Int) :
    (HeapCalls.scalarFunction Solvers.P46Ast.module "solve" [.int after]).Equiv
      (Calls.function P46Ast.module "naive" [.int after]) := by
  by_cases success : Success after
  · obtain ⟨_, optimized⟩ := result after success []
    have left := optimized.scalar_finishes (values := [.int after])
    have right := (data.«project-euler-statements».data.py.project_euler_statements.p46.naive.result after success).finishes
    intro outcome
    constructor
    · intro finished
      rw [finished.unique left]
      exact right
    · intro finished
      rw [finished.unique right]
      exact left
  · have left := diverges after success
    have right := data.«project-euler-statements».data.py.project_euler_statements.p46.naive.diverges after success
    intro outcome
    constructor
    · intro finished
      exact False.elim (Execution.diverges_not_halts left ⟨outcome, finished⟩)
    · intro finished
      exact False.elim (Execution.diverges_not_halts right ⟨outcome, finished⟩)

theorem supported (after : Int) :
    (HeapCalls.scalarFunction Solvers.P46Ast.module "solve" [.int after]).Supported ∧
      (Calls.function P46Ast.module "naive" [.int after]).Supported := by
  by_cases success : Success after
  · obtain ⟨_, optimized⟩ := result after success []
    exact ⟨(optimized.scalar_returns (values := [.int after])).supported,
      (data.«project-euler-statements».data.py.project_euler_statements.p46.naive.result after success).returns.supported⟩
  · exact ⟨(diverges after success).supported,
      (data.«project-euler-statements».data.py.project_euler_statements.p46.naive.diverges after success).supported⟩

theorem «Supported» :
    (∀ after : Int, (HeapCalls.scalarFunction Solvers.P46Ast.module "solve" [.int after]).Supported) ∧
      (∀ after : Int, (Calls.function P46Ast.module "naive" [.int after]).Supported) :=
  ⟨fun after => (supported after).1, fun after => (supported after).2⟩

/-- Termination is characterized by existence, rather than an externally supplied search bound. -/
theorem returns_iff (after : Int) :
    (HeapCalls.scalarFunction Solvers.P46Ast.module "solve" [.int after]).Returns ↔ Success after := by
  constructor
  · rintro ⟨value, returned⟩
    by_contra absent
    exact Execution.diverges_not_halts (diverges after absent) ⟨.ok value, returned⟩
  · intro success
    obtain ⟨_, returned⟩ := result after success []
    exact returned.scalar_returns (values := [.int after])

theorem reference_returns_iff (after : Int) :
    (Calls.function P46Ast.module "naive" [.int after]).Returns ↔ Success after :=
  (Execution.Equiv.returns_iff («Equiv(naive)» after)).symm.trans (returns_iff after)

/-- The known explicit counterexample supplies a termination witness for the default search. -/
theorem default_success : Success 0 := by
  refine ⟨2884, ?_⟩
  have numeral : value 2884 = ProjectEulerSolutions.PrimeTwiceSquareSearch.winner := by
    norm_num [value, ProjectEulerSolutions.PrimeTwiceSquareSearch.winner]
  have selected := accepts_of 0 ProjectEulerSolutions.PrimeTwiceSquareSearch.winner
    (by norm_num [ProjectEulerSolutions.PrimeTwiceSquareSearch.winner])
    ProjectEulerSolutions.PrimeTwiceSquareSearch.winner_accepted
  exact (congrArg (fun n => accepts 0 n = true) numeral).mpr selected

theorem default_returns : (HeapCalls.scalarFunction Solvers.P46Ast.module "solve" []).Returns := by
  obtain ⟨finalHeap, returned⟩ := result 0 default_success []
  have defaultRun : HeapCalls.ReturnsValue Solvers.P46Ast.module "solve" [] []
      (.scalar (.int (answer 0 default_success))) finalHeap :=
    HeapCalls.ReturnsValue.with_defaults rfl rfl rfl returned
  exact defaultRun.scalar_returns (values := [])

end solvers.«46».solve
