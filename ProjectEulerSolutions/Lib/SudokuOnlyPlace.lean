import ProjectEulerSolutions.Lib.SudokuPropagation

/-! A missing digit makes a unit impossible; a unique location forces that digit. -/
namespace ProjectEulerSolutions.Sudoku.OnlyPlace
set_option Elab.async false
noncomputable section

def places (candidates : Candidates) (cells : List Cell) (digit : Digit) : List Cell :=
  cells.filter fun cell => decide (digit ∈ candidates cell)

theorem mem_places (candidates : Candidates) (cells : List Cell) (digit : Digit) (cell : Cell) :
    cell ∈ places candidates cells digit ↔ cell ∈ cells ∧ digit ∈ candidates cell := by simp [places]

theorem empty (candidates : Candidates) (unit : Fin 9 → Cell) (digit : Digit)
    (none : places candidates ((List.finRange 9).map unit) digit = []) :
    ∀ index, digit ∉ candidates (unit index) := by
  intro index present
  have member := (mem_places candidates ((List.finRange 9).map unit) digit (unit index)).mpr ⟨by simp, present⟩
  rw [none] at member
  contradiction

theorem singleton (candidates : Candidates) (unit : Fin 9 → Cell) (digit : Digit) (cell : Cell)
    (one : places candidates ((List.finRange 9).map unit) digit = [cell]) :
    digit ∈ candidates cell ∧ (∀ index, digit ∈ candidates (unit index) → unit index = cell) := by
  constructor
  · exact ((mem_places candidates ((List.finRange 9).map unit) digit cell).mp (by rw [one]; simp)).2
  · intro index present
    have member := (mem_places candidates ((List.finRange 9).map unit) digit (unit index)).mpr ⟨by simp, present⟩
    rw [one] at member
    exact List.mem_singleton.mp member

def force (state : Propagation.State) (cell : Cell) (digit : Digit) : Propagation.State :=
  { state with candidates := Function.update state.candidates cell {digit}, changed := true }

def inspect (state : Propagation.State) (cells : List Cell) (digit : Digit) : Option Propagation.State :=
  match places state.candidates cells digit with
  | [] => none
  | [cell] => if state.candidates cell = {digit} then some state else some (force state cell digit)
  | _ :: _ :: _ => some state

theorem force_subset (state : Propagation.State) (cell : Cell) (digit : Digit)
    (present : digit ∈ state.candidates cell) (other : Cell) :
    (force state cell digit).candidates other ⊆ state.candidates other := by
  by_cases same : other = cell
  · subst other
    simpa [force] using Finset.singleton_subset_iff.mpr present
  · simp [force, Function.update_of_ne same]

theorem force_nonempty (state : Propagation.State) (cell : Cell) (digit : Digit)
    (nonempty : Propagation.Nonempty state) : Propagation.Nonempty (force state cell digit) := by
  intro other
  by_cases same : other = cell
  · subst other
    simp [force]
  · simpa [force, Function.update_of_ne same] using nonempty other

theorem force_weight (state : Propagation.State) (cell : Cell) (digit : Digit)
    (present : digit ∈ state.candidates cell) (different : state.candidates cell ≠ {digit}) :
    weight (force state cell digit).candidates < weight state.candidates := by
  apply branch_weight_lt
  have subset : {digit} ⊂ state.candidates cell :=
    Finset.ssubset_iff_subset_ne.mpr ⟨Finset.singleton_subset_iff.mpr present, Ne.symm different⟩
  simpa using Finset.card_lt_card subset

theorem force_progress (state : Propagation.State) (cell : Cell) (digit : Digit)
    (present : digit ∈ state.candidates cell) (different : state.candidates cell ≠ {digit}) :
    Propagation.Progress state (force state cell digit) := by
  have smaller := force_weight state cell digit present different
  exact ⟨force_subset state cell digit present, Nat.add_le_add_right smaller.le state.queued.length,
    smaller.le, fun _ => Or.inr smaller, fun _ => rfl⟩

theorem force_equiv (state : Propagation.State) (unit : Fin 9 → Cell) (digit : Digit) (cell : Cell)
    (peers : ∀ i j, i ≠ j → Peer (unit i) (unit j))
    (one : places state.candidates ((List.finRange 9).map unit) digit = [cell])
    (assignment : Assignment) (legal : Legal assignment) :
    Admits (force state cell digit).candidates assignment ↔ Admits state.candidates assignment := by
  obtain ⟨present, only⟩ := singleton state.candidates unit digit cell one
  exact force_only_place_iff state.candidates assignment legal unit peers cell digit present only

theorem inspect_none (state : Propagation.State) (cells : List Cell) (digit : Digit) :
    inspect state cells digit = none ↔ places state.candidates cells digit = [] := by
  unfold inspect
  split
  · simp_all
  · split <;> simp_all
  · simp_all

theorem inspect_impossible (state : Propagation.State) (unit : Fin 9 → Cell) (digit : Digit)
    (peers : ∀ i j, i ≠ j → Peer (unit i) (unit j))
    (none : inspect state ((List.finRange 9).map unit) digit = none) :
    ¬ ∃ assignment, Legal assignment ∧ Admits state.candidates assignment :=
  missing_digit_impossible state.candidates unit peers digit
    (empty state.candidates unit digit ((inspect_none state _ digit).mp none))

theorem inspect_some (state after : Propagation.State) (unit : Fin 9 → Cell) (digit : Digit)
    (peers : ∀ i j, i ≠ j → Peer (unit i) (unit j)) (nonempty : Propagation.Nonempty state)
    (done : inspect state ((List.finRange 9).map unit) digit = some after) :
    Propagation.Progress state after ∧ Propagation.Nonempty after ∧ after.queued = state.queued ∧
      (∀ assignment, Legal assignment → (Admits after.candidates assignment ↔ Admits state.candidates assignment)) := by
  unfold inspect at done
  split at done
  · contradiction
  · rename_i cell one
    split at done
    · cases done
      exact ⟨.refl _, nonempty, rfl, fun _ _ => Iff.rfl⟩
    · rename_i different
      cases done
      exact ⟨force_progress state cell digit (singleton state.candidates unit digit cell one).1 different,
        force_nonempty state cell digit nonempty, rfl, fun assignment legal => force_equiv state unit digit cell peers one assignment legal⟩
  · cases done
    exact ⟨.refl _, nonempty, rfl, fun _ _ => Iff.rfl⟩

structure Success (before after : Propagation.State) : Prop where
  progress : Propagation.Progress before after
  nonempty : Propagation.Nonempty after
  queued : after.queued = before.queued
  equivalent : ∀ assignment, Legal assignment →
    (Admits after.candidates assignment ↔ Admits before.candidates assignment)
  stable : after.changed = false → after.candidates = before.candidates

theorem unchanged_before {before after : Propagation.State} (progress : Propagation.Progress before after)
    (unchanged : after.changed = false) : before.changed = false := by
  cases changed : before.changed with
  | false => rfl
  | true => have := progress.retainsChanged changed; simp [unchanged] at this

theorem inspect_unchanged (before after : Propagation.State) (cells : List Cell) (digit : Digit)
    (done : inspect before cells digit = some after) (unchanged : after.changed = false) :
    after = before ∧ places after.candidates cells digit ≠ [] := by
  have available : places before.candidates cells digit ≠ [] := by
    intro missing
    have rejected := (inspect_none before cells digit).mpr missing
    rw [done] at rejected
    contradiction
  unfold inspect at done
  split at done
  · contradiction
  · split at done
    · cases done
      exact ⟨rfl, available⟩
    · cases done
      simp [force] at unchanged
  · cases done
    exact ⟨rfl, available⟩

def Covers (candidates : Candidates) (unit : Fin 9 → Cell) (digit : Digit) : Prop :=
  ∃ index, digit ∈ candidates (unit index)

theorem covers_iff (candidates : Candidates) (unit : Fin 9 → Cell) (digit : Digit) :
    Covers candidates unit digit ↔ places candidates ((List.finRange 9).map unit) digit ≠ [] := by
  simp [Covers, places, List.filter_eq_nil_iff]

def Checked (result : Option Propagation.State) (property : Candidates → Prop) : Prop :=
  match result with
  | none => True
  | some after => after.changed = false → property after.candidates

theorem Success.refl (state : Propagation.State) (nonempty : Propagation.Nonempty state) : Success state state :=
  ⟨.refl _, nonempty, rfl, fun _ _ => Iff.rfl, fun _ => rfl⟩

theorem Success.trans {a b c : Propagation.State} (first : Success a b) (second : Success b c) : Success a c :=
  ⟨first.progress.trans second.progress, second.nonempty, second.queued.trans first.queued,
    fun assignment legal => (second.equivalent assignment legal).trans (first.equivalent assignment legal),
    fun unchanged => (second.stable unchanged).trans (first.stable (unchanged_before second.progress unchanged))⟩

def Conclusion (before : Propagation.State) (result : Option Propagation.State) : Prop :=
  match result with
  | none => ¬ ∃ assignment, Legal assignment ∧ Admits before.candidates assignment
  | some after => Success before after

theorem conclusion (state : Propagation.State) (unit : Fin 9 → Cell) (digit : Digit)
    (peers : ∀ i j, i ≠ j → Peer (unit i) (unit j)) (nonempty : Propagation.Nonempty state) :
    Conclusion state (inspect state ((List.finRange 9).map unit) digit) := by
  cases result : inspect state ((List.finRange 9).map unit) digit with
  | none => exact inspect_impossible state unit digit peers result
  | some after =>
      obtain ⟨progress, nonempty, queued, equivalent⟩ := inspect_some state after unit digit peers nonempty result
      exact ⟨progress, nonempty, queued, equivalent, fun unchanged =>
        congrArg Propagation.State.candidates (inspect_unchanged state after _ digit result unchanged).1⟩

theorem Conclusion.trans {a b : Propagation.State} {result : Option Propagation.State}
    (first : Success a b) (second : Conclusion b result) : Conclusion a result := by
  cases result with
  | none =>
      rintro ⟨assignment, legal, admitted⟩
      exact second ⟨assignment, legal, (first.equivalent assignment legal).mpr admitted⟩
  | some after => exact first.trans second

end
end ProjectEulerSolutions.Sudoku.OnlyPlace
