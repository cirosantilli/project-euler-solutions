import ProjectEulerSolutions.Lib.SudokuSearch
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-! Candidate propagation preserves Sudoku completions and strictly decreases a finite measure. -/
namespace ProjectEulerSolutions.Sudoku

set_option Elab.async false

abbrev Candidates := Cell → Finset Digit
abbrev Assignment := Cell → Digit

def Admits (candidates : Candidates) (assignment : Assignment) : Prop :=
  ∀ cell, assignment cell ∈ candidates cell

def Legal (assignment : Assignment) : Prop :=
  ∀ a b, Peer a b → assignment a ≠ assignment b

def initialCandidates (board : Board) : Candidates := fun cell =>
  match board cell with
  | none => Finset.univ
  | some digit => {digit}

theorem initial_admits_iff (board : Board) (assignment : Assignment) :
    Admits (initialCandidates board) assignment ↔
      Extends board (fun cell => some (assignment cell)) := by
  constructor
  · intro admitted cell digit found
    have member := admitted cell
    simpa [initialCandidates, found] using member
  · intro extension cell
    cases found : board cell with
    | none => simp [initialCandidates, found]
    | some digit =>
        have same := extension cell digit found
        simpa [initialCandidates, found] using same

theorem legal_full (assignment : Assignment) (legal : Legal assignment) :
    Valid (fun cell => some (assignment cell)) ∧ Full (fun cell => some (assignment cell)) := by
  refine ⟨?_, by simp [Full]⟩
  intro a b peer digit first second
  simp only [Option.some.injEq] at first second
  exact legal a b peer (first.trans second.symm)

theorem admits_subset {before after : Candidates} (subset : ∀ cell, after cell ⊆ before cell)
    {assignment : Assignment} (admitted : Admits after assignment) : Admits before assignment :=
  fun cell => subset cell (admitted cell)

/-- A known singleton excludes its digit from every peer, without losing any legal assignment. -/
theorem erase_peer_iff (candidates : Candidates) (assignment : Assignment) (legal : Legal assignment)
    {cell other : Cell} {digit : Digit} (single : candidates cell = {digit}) (peer : Peer cell other) :
    Admits (Function.update candidates other ((candidates other).erase digit)) assignment ↔
      Admits candidates assignment := by
  constructor
  · apply admits_subset
    intro current
    by_cases same : current = other
    · subst current
      simpa using Finset.erase_subset digit (candidates other)
    · simp [Function.update_of_ne same]
  · intro admitted current
    by_cases same : current = other
    · subst current
      simp only [Function.update_self, Finset.mem_erase]
      have atCell : assignment cell = digit := by simpa [single] using admitted cell
      exact ⟨fun equal => legal cell other peer (atCell.trans equal.symm), admitted other⟩
    · simpa [Function.update_of_ne same] using admitted current

/-- A digit with only one possible location in a unit is forced there. -/
theorem force_only_place_iff (candidates : Candidates) (assignment : Assignment) (legal : Legal assignment)
    (unit : Fin 9 → Cell) (peers : ∀ i j, i ≠ j → Peer (unit i) (unit j))
    (cell : Cell) (digit : Digit) (available : digit ∈ candidates cell)
    (only : ∀ index, digit ∈ candidates (unit index) → unit index = cell) :
    Admits (Function.update candidates cell {digit}) assignment ↔ Admits candidates assignment := by
  constructor
  · apply admits_subset
    intro current
    by_cases same : current = cell
    · subst current
      simpa using Finset.singleton_subset_iff.mpr available
    · simp [Function.update_of_ne same]
  · intro admitted current
    by_cases same : current = cell
    · subst current
      obtain ⟨valid, full⟩ := legal_full assignment legal
      obtain ⟨index, found⟩ := unit_contains_every_digit valid full unit peers digit
      have equal : assignment (unit index) = digit := Option.some.inj found
      have place : unit index = cell := only index (equal ▸ admitted (unit index))
      simp only [Function.update_self, Finset.mem_singleton]
      simpa [place] using equal
    · simpa [Function.update_of_ne same] using admitted current

theorem empty_cell_impossible (candidates : Candidates) {cell : Cell} (empty : candidates cell = ∅) :
    ¬ ∃ assignment, Admits candidates assignment := by
  rintro ⟨assignment, admitted⟩
  have member := admitted cell
  simp [empty] at member

theorem missing_digit_impossible (candidates : Candidates)
    (unit : Fin 9 → Cell) (peers : ∀ i j, i ≠ j → Peer (unit i) (unit j))
    (digit : Digit) (missing : ∀ index, digit ∉ candidates (unit index)) :
    ¬ ∃ assignment, Legal assignment ∧ Admits candidates assignment := by
  rintro ⟨assignment, legal, admitted⟩
  obtain ⟨valid, full⟩ := legal_full assignment legal
  obtain ⟨index, found⟩ := unit_contains_every_digit valid full unit peers digit
  have equal : assignment (unit index) = digit := Option.some.inj found
  exact missing index (equal ▸ admitted (unit index))

def weight (candidates : Candidates) : Nat := ∑ cell, (candidates cell).card

theorem weight_update_lt (candidates : Candidates) (cell : Cell) (replacement : Finset Digit)
    (smaller : replacement.card < (candidates cell).card) :
    weight (Function.update candidates cell replacement) < weight candidates := by
  apply Finset.sum_lt_sum
  · intro current _
    by_cases same : current = cell
    · subst current
      simpa using smaller.le
    · simp [Function.update_of_ne same]
  · exact ⟨cell, Finset.mem_univ _, by simpa using smaller⟩

theorem erase_weight_lt (candidates : Candidates) (cell : Cell) (digit : Digit)
    (present : digit ∈ candidates cell) :
    weight (Function.update candidates cell ((candidates cell).erase digit)) < weight candidates :=
  weight_update_lt candidates cell _ (Finset.card_erase_lt_of_mem present)

/-- A branch at a cell with multiple candidates strictly decreases the total candidate count. -/
theorem branch_weight_lt (candidates : Candidates) (cell : Cell) (digit : Digit)
    (multiple : 1 < (candidates cell).card) :
    weight (Function.update candidates cell {digit}) < weight candidates :=
  weight_update_lt candidates cell _ (by simpa using multiple)

theorem branch_complete (candidates : Candidates) (assignment : Assignment) (cell : Cell) :
    Admits candidates assignment ↔
      ∃ digit ∈ candidates cell, Admits (Function.update candidates cell {digit}) assignment := by
  constructor
  · intro admitted
    refine ⟨assignment cell, admitted cell, ?_⟩
    intro current
    by_cases same : current = cell
    · subst current
      simp
    · simpa [Function.update_of_ne same] using admitted current
  · rintro ⟨digit, present, admitted⟩
    apply admits_subset _ admitted
    intro current
    by_cases same : current = cell
    · subst current
      simpa using Finset.singleton_subset_iff.mpr present
    · simp [Function.update_of_ne same]

end ProjectEulerSolutions.Sudoku
