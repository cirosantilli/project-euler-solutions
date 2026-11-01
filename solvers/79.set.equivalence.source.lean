import solvers.«79.set.source»
import data.«project-euler-statements».data.py.project_euler_statements.«p79.set.source»
import lang.py.FormalSW.IteratorLifting

/-! Finite string sets have matching possible behaviors, including order-sensitive validation failures. -/
namespace solvers.«79».source
open FormalSW.PyAst HeapCalls
open solvers.«79».graph solvers.«79».topo
open data.«project-euler-statements».data.py.project_euler_statements
set_option Elab.async false

theorem validated_outcome_not_type_error (checked attempts : List String) :
    p79.validatedOutcome checked attempts ≠ .error ⟨"TypeError", []⟩ := by
  induction attempts generalizing checked with
  | nil =>
      simp only [p79.validatedOutcome, p79.checkedOutcome]
      split <;> simp_all [p79.cycleException]
  | cons s rest ih =>
      simp only [p79.validatedOutcome]
      split
      · exact ih _
      · simp [p79.invalidException]

/-- Both actual calls finish for every schedule; their private allocations preserve the caller's heap. -/
theorem set_calls_finish (choices : ChoiceSchedule) (base : Mutable.Heap)
    (address : Nat) (attempts : List String)
    (found : base[address]? = some (.set (attempts.map Constant.string))) :
    ∃ left right,
      ((IteratorCalls.call program "solve" [.ref address] base).run choices).Finishes
        ⟨solveOutcome (SetCalls.selectionOrder choices 0 attempts), [], left, base.length⟩ ∧
      ((IteratorCalls.call p79.program "naive" [.ref address] base).run choices).Finishes
        ⟨solveOutcome (SetCalls.selectionOrder choices 0 attempts), [], right, base.length⟩ ∧
      base <+: left ∧ base <+: right := by
  obtain ⟨left, optimized, leftSaved⟩ := solve_set_finishes choices base address attempts found
  obtain ⟨right, reference, rightSaved⟩ := p79.naive_set_finishes choices base address attempts found
  rw [← solve_outcome_eq_reference] at reference
  have safe : solveOutcome (SetCalls.selectionOrder choices 0 attempts) ≠ .error ⟨"TypeError", []⟩ := by
    rw [solve_outcome_eq_reference]
    exact validated_outcome_not_type_error _ _
  refine ⟨left, right, ?_, ?_, leftSaved, rightSaved⟩
  · exact IteratorCalls.finishes_of_set optimized (by intro completion equal; cases equal; exact safe)
  · exact IteratorCalls.finishes_of_set reference (by intro completion equal; cases equal; exact safe)

theorem set_behavior_equiv (choices : ChoiceSchedule) (base : Mutable.Heap)
    (address : Nat) (attempts : List String) (valid : base.Valid)
    (found : base[address]? = some (.set (attempts.map Constant.string))) :
    ((IteratorCalls.call program "solve" [.ref address] base).run choices).BehaviorEquiv
      ((IteratorCalls.call p79.program "naive" [.ref address] base).run choices) := by
  obtain ⟨left, right, optimized, reference, leftSaved, rightSaved⟩ :=
    set_calls_finish choices base address attempts found
  exact Mutable.Call.BehaviorEquiv.of_finishes optimized reference
    (outcome_allocations base left right valid leftSaved rightSaved
      (SetCalls.selectionOrder choices 0 attempts))

theorem set_possible_equiv (base : Mutable.Heap) (address : Nat) (attempts : List String)
    (valid : base.Valid) (found : base[address]? = some (.set (attempts.map Constant.string))) :
    (IteratorCalls.call program "solve" [.ref address] base).PossibleEquiv
      (IteratorCalls.call p79.program "naive" [.ref address] base) :=
  Mutable.CallFamily.PossibleEquiv.of_pointwise
    (fun choices => set_behavior_equiv choices base address attempts valid found)

/-- Coverage is established separately from equivalence: neither family has an unsupported or divergent run. -/
theorem set_halts (base : Mutable.Heap) (address : Nat) (attempts : List String)
    (found : base[address]? = some (.set (attempts.map Constant.string))) :
    (∀ choices, ((IteratorCalls.call program "solve" [.ref address] base).run choices).Halts) ∧
      (∀ choices, ((IteratorCalls.call p79.program "naive" [.ref address] base).run choices).Halts) := by
  constructor <;> intro choices
  · obtain ⟨left, _, optimized, _, _, _⟩ := set_calls_finish choices base address attempts found
    exact ⟨_, optimized⟩
  · obtain ⟨_, right, _, reference, _, _⟩ := set_calls_finish choices base address attempts found
    exact ⟨_, reference⟩

theorem valid_set_outcome (choices : ChoiceSchedule) (attempts : List String)
    (valid : ∀ s ∈ attempts, ValidAttempt s) :
    solveOutcome (SetCalls.selectionOrder choices 0 attempts) = solveOutcome attempts := by
  have permutation := SetCalls.selectionOrder_perm choices 0 attempts
  have selectedValid : ∀ s ∈ SetCalls.selectionOrder choices 0 attempts,
      s.length = 3 ∧ Strings.isDigit s = true :=
    fun s member => valid s (permutation.mem_iff.mp member)
  rw [solve_outcome_eq_reference, solve_outcome_eq_reference,
    p79.validatedOutcome_valid _ selectedValid, p79.validatedOutcome_valid _ valid]
  exact p79.checkedOutcome_perm _ _ permutation

/-- Valid attempts give guaranteed equivalence even with independent input and internal graph orders. -/
theorem valid_set_equiv (base : Mutable.Heap) (address : Nat) (attempts : List String)
    (heapValid : base.Valid) (found : base[address]? = some (.set (attempts.map Constant.string)))
    (valid : ∀ s ∈ attempts, ValidAttempt s) :
    (IteratorCalls.call program "solve" [.ref address] base).Equiv
      (IteratorCalls.call p79.program "naive" [.ref address] base) := by
  intro leftChoices rightChoices
  obtain ⟨left, _, optimized, _, leftSaved, _⟩ := set_calls_finish leftChoices base address attempts found
  obtain ⟨_, right, _, reference, _, rightSaved⟩ := set_calls_finish rightChoices base address attempts found
  rw [valid_set_outcome leftChoices attempts valid] at optimized
  rw [valid_set_outcome rightChoices attempts valid] at reference
  exact Mutable.Call.equiv_of_finishes optimized reference
    (outcome_allocations base left right heapValid leftSaved rightSaved attempts)

end solvers.«79».source
