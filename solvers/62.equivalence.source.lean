import solvers.«62.unbounded.source»

/-! Unconditional equivalence for the shared class-size argument, including absent classes. -/
set_option Elab.async false
namespace solvers.«62».source_equivalence
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerSolutions.CubicSearch ProjectEulerSolutions.CubicWidthBounds
open unbounded
private abbrev program := Solvers.P62Ast.module
private abbrev number (n : Int) : Mutable.Value := .scalar (.int n)

def NoReturn (target : Int) (state : ProjectEulerSolutions.CubicSearch.State) : Prop :=
  ∀ answer, ¬ ProjectEulerSolutions.CubicSearch.Returns target state answer

/-- Without a finite mathematical witness the optimized source keeps executing supported iterations. -/
theorem optimized_diverges (target : Int) (absent : NoReturn target start) :
    (HeapCalls.scalarFunction program "solve" [.int target]).Diverges := by
  let args := ((findFunctionIn program.body "solve").getD ([], [])).1
  let sourceBody := ((findFunctionIn program.body "solve").getD ([], [])).2
  let entry := functionEnv args sourceBody [number target] []
  let invariant : HeapCalls.Env → Mutable.Heap → Prop := fun env heap =>
    (env = search.initialEnv entry [] ∧ heap = search.initialHeap []) ∨
      ∃ state cells, search.Invariant heap env target state 0 cells 0 ∧ NoReturn target state
  apply scalar_while_diverges program "solve" [.int target] args sourceBody search.initialization
    groups.iteration [number target] rfl rfl search.code_source rfl invariant
    (search.initialEnv entry []) (search.initialHeap [])
  · exact search.initialize_result [] entry
  · exact Or.inl ⟨rfl, rfl⟩
  · intro env heap current
    rcases current with ⟨rfl, rfl⟩ | ⟨state, cells, represented, noReturn⟩
    · obtain ⟨finalHeap, finalEnv, finalCells, advanced, represented, _⟩ :=
        search.first_iteration target [] entry (by rfl)
      refine ⟨finalEnv, finalHeap, advanced, Or.inr ⟨ready, finalCells, represented, ?_⟩⟩
      intro answer returned
      exact absent answer (.next (step_start target) returned)
    · have executed := search.step_result target state heap env 0 cells 0 represented
      cases checked : ProjectEulerSolutions.CubicSearch.step target state with
      | inl value => exact False.elim (noReturn value (.done checked))
      | inr following =>
          rw [checked] at executed
          obtain ⟨finalHeap, finalEnv, finalCells, advanced, preserved, _, _⟩ := executed
          refine ⟨finalEnv, finalHeap, advanced, Or.inr ⟨following, finalCells, preserved, ?_⟩⟩
          intro answer returned
          exact noReturn answer (.next checked returned)

theorem no_model_return (target : Int)
    (absent : ∀ digits, 1 ≤ digits → widthAnswer target digits = 0) : NoReturn target start := by
  intro answer returned
  obtain ⟨digits, positive, success⟩ := returns_has_width
    (ProjectEulerSolutions.CubicWindows.invariant_start target) returned
  exact success (by rw [width_choice, absent digits positive])

/-- A first successful width returns the same scalar from both complete source calls. -/
theorem common_result (target : Int) (digits : Nat) (positive : 1 ≤ digits)
    (success : widthAnswer target digits ≠ 0)
    (earlier : ∀ width, 1 ≤ width → width < digits → widthAnswer target width = 0) :
    ∃ finalHeap extra,
      ReturnsValue program "solve" [number target] [] (number (widthAnswer target digits)) finalHeap ∧
      ReturnsValue P62Ast.module "naive" [number target] [] (number (widthAnswer target digits)) extra := by
  have closed := ProjectEulerSolutions.CubicWidthBounds.closed digits positive
  obtain ⟨finalHeap, optimized, _⟩ := search.result target _ []
    (closed_window.witness (stop digits - 1) target
      (by simpa only [Nat.sub_add_cancel (stop_positive digits)] using closed) success)
  obtain ⟨extra, reference⟩ := reference_result target digits positive success earlier []
  exact ⟨finalHeap, extra, optimized, reference⟩

private theorem least_width (target : Int)
    (existsWidth : ∃ digits : Nat, 1 ≤ digits ∧ widthAnswer target digits ≠ 0) :
    ∃ digits : Nat, 1 ≤ digits ∧ widthAnswer target digits ≠ 0 ∧
      ∀ width, 1 ≤ width → width < digits → widthAnswer target width = 0 := by
  refine ⟨Nat.find existsWidth, (Nat.find_spec existsWidth).1, (Nat.find_spec existsWidth).2, ?_⟩
  intro width positive before
  by_contra success
  exact Nat.find_min existsWidth before ⟨positive, success⟩

/-- Both sources return the same value or both diverge, for every signed class size. -/
theorem equivalent (target : Int) :
    (HeapCalls.scalarFunction program "solve" [.int target]).Equiv
      (HeapCalls.scalarFunction P62Ast.module "naive" [.int target]) := by
  by_cases existsWidth : ∃ digits : Nat, 1 ≤ digits ∧ widthAnswer target digits ≠ 0
  · obtain ⟨digits, positive, success, earlier⟩ := least_width target existsWidth
    obtain ⟨_, _, optimized, reference⟩ := common_result target digits positive success earlier
    have left := optimized.scalar_finishes (values := [.int target])
    have right := reference.scalar_finishes (values := [.int target])
    intro outcome
    constructor
    · intro finished
      rw [finished.unique left]
      exact right
    · intro finished
      rw [finished.unique right]
      exact left
  · have absent : ∀ digits, 1 ≤ digits → widthAnswer target digits = 0 := by
      intro digits positive
      by_contra success
      exact existsWidth ⟨digits, positive, success⟩
    have left := optimized_diverges target (no_model_return target absent)
    have right := reference_diverges target absent
    intro outcome
    constructor
    · intro finished
      exact False.elim (Execution.diverges_not_halts left ⟨outcome, finished⟩)
    · intro finished
      exact False.elim (Execution.diverges_not_halts right ⟨outcome, finished⟩)

theorem supported (target : Int) :
    (HeapCalls.scalarFunction program "solve" [.int target]).Supported ∧
      (HeapCalls.scalarFunction P62Ast.module "naive" [.int target]).Supported := by
  by_cases existsWidth : ∃ digits : Nat, 1 ≤ digits ∧ widthAnswer target digits ≠ 0
  · obtain ⟨digits, positive, success, earlier⟩ := least_width target existsWidth
    obtain ⟨_, _, optimized, reference⟩ := common_result target digits positive success earlier
    exact ⟨(optimized.scalar_returns (values := [.int target])).supported,
      (reference.scalar_returns (values := [.int target])).supported⟩
  · have absent : ∀ digits, 1 ≤ digits → widthAnswer target digits = 0 := by
      intro digits positive
      by_contra success
      exact existsWidth ⟨digits, positive, success⟩
    exact ⟨(optimized_diverges target (no_model_return target absent)).supported,
      (reference_diverges target absent).supported⟩

/-- Termination is exactly the existence of a successful complete width, with no target restriction. -/
theorem returns_iff_width (target : Int) :
    (HeapCalls.scalarFunction program "solve" [.int target]).Returns ↔
      ∃ digits : Nat, 1 ≤ digits ∧ widthAnswer target digits ≠ 0 := by
  constructor
  · rintro ⟨value, returned⟩
    by_contra noWidth
    have absent : ∀ digits, 1 ≤ digits → widthAnswer target digits = 0 := by
      intro digits positive
      by_contra success
      exact noWidth ⟨digits, positive, success⟩
    exact Execution.diverges_not_halts (optimized_diverges target (no_model_return target absent))
      ⟨.ok value, returned⟩
  · intro existsWidth
    obtain ⟨digits, positive, success, earlier⟩ := least_width target existsWidth
    obtain ⟨_, _, optimized, _⟩ := common_result target digits positive success earlier
    exact optimized.scalar_returns (values := [.int target])

theorem reference_returns_iff_width (target : Int) :
    (HeapCalls.scalarFunction P62Ast.module "naive" [.int target]).Returns ↔
      ∃ digits : Nat, 1 ≤ digits ∧ widthAnswer target digits ≠ 0 :=
  (Execution.Equiv.returns_iff (equivalent target)).symm.trans (returns_iff_width target)

end solvers.«62».source_equivalence

namespace solvers.«62».solve
open FormalSW.PyAst

theorem «Equiv(naive)» (target : Int) :
    (HeapCalls.scalarFunction Solvers.P62Ast.module "solve" [.int target]).Equiv
      (HeapCalls.scalarFunction P62Ast.module "naive" [.int target]) := source_equivalence.equivalent target

theorem «Supported» :
    (∀ target : Int, (HeapCalls.scalarFunction Solvers.P62Ast.module "solve" [.int target]).Supported) ∧
      (∀ target : Int, (HeapCalls.scalarFunction P62Ast.module "naive" [.int target]).Supported) :=
  ⟨fun target => (source_equivalence.supported target).1, fun target => (source_equivalence.supported target).2⟩

end solvers.«62».solve
