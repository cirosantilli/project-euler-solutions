import ProjectEulerSolutions.Lib.SudokuCandidates

/-! Singleton propagation over arbitrary finite peer orders. -/
namespace ProjectEulerSolutions.Sudoku.Propagation
set_option Elab.async false
noncomputable section

structure State where
  candidates : Candidates
  queued : List Cell
  changed : Bool

def Nonempty (state : State) : Prop := ∀ cell, state.candidates cell ≠ ∅
def Singles (state : State) : Prop := ∀ cell ∈ state.queued, (state.candidates cell).card = 1
def potential (state : State) : Nat := weight state.candidates + state.queued.length

def reduced (state : State) (other : Cell) (digit : Digit) : State :=
  let remaining := (state.candidates other).erase digit
  ⟨Function.update state.candidates other remaining,
    state.queued ++ if remaining.card = 1 then [other] else [], true⟩

def remove (state : State) (other : Cell) (digit : Digit) : Option State :=
  if digit ∈ state.candidates other then
    if (state.candidates other).erase digit = ∅ then none else some (reduced state other digit)
  else some state

def peers (state : State) (digit : Digit) : List Cell → Option State
  | [] => some state
  | other :: rest => (remove state other digit).bind fun after => peers after digit rest

theorem remove_none (state : State) (other : Cell) (digit : Digit) :
    remove state other digit = none ↔ digit ∈ state.candidates other ∧ (state.candidates other).erase digit = ∅ := by
  by_cases present : digit ∈ state.candidates other <;>
    by_cases empty : (state.candidates other).erase digit = ∅ <;> simp [remove, present, empty]

theorem remove_some (state after : State) (other : Cell) (digit : Digit) :
    remove state other digit = some after ↔
      (digit ∉ state.candidates other ∧ after = state) ∨
      (digit ∈ state.candidates other ∧ (state.candidates other).erase digit ≠ ∅ ∧ after = reduced state other digit) := by
  by_cases present : digit ∈ state.candidates other <;>
    by_cases empty : (state.candidates other).erase digit = ∅ <;> simp [remove, present, empty, eq_comm]

theorem reduced_subset (state : State) (other : Cell) (digit : Digit) (cell : Cell) :
    (reduced state other digit).candidates cell ⊆ state.candidates cell := by
  by_cases same : cell = other
  · subst cell
    simpa [reduced] using Finset.erase_subset digit (state.candidates other)
  · simp [reduced, Function.update_of_ne same]

theorem reduced_nonempty (state : State) (other : Cell) (digit : Digit)
    (before : Nonempty state) (remaining : (state.candidates other).erase digit ≠ ∅) :
    Nonempty (reduced state other digit) := by
  intro cell
  by_cases same : cell = other
  · subst cell
    simpa [reduced] using remaining
  · simpa [reduced, Function.update_of_ne same] using before cell

theorem reduced_singles (state : State) (other : Cell) (digit : Digit)
    (before : Singles state) (remaining : (state.candidates other).erase digit ≠ ∅) :
    Singles (reduced state other digit) := by
  intro cell member
  rcases List.mem_append.mp member with old | added
  · by_cases same : cell = other
    · subst cell
      have size := before other old
      have positive := Finset.card_pos.mpr (Finset.nonempty_iff_ne_empty.mpr remaining)
      have bounded := Finset.card_le_card (Finset.erase_subset digit (state.candidates other))
      simp only [reduced, Function.update_self]
      omega
    · simpa [reduced, Function.update_of_ne same] using before cell old
  · split at added
    · have same : cell = other := List.mem_singleton.mp added
      subst cell
      simpa [reduced] using ‹((state.candidates other).erase digit).card = 1›
    · contradiction

theorem reduced_weight (state : State) (other : Cell) (digit : Digit) (present : digit ∈ state.candidates other) :
    weight (reduced state other digit).candidates < weight state.candidates :=
  erase_weight_lt state.candidates other digit present

/-- Each appended singleton consumes at least one candidate, so the queue cannot grow without progress. -/
theorem reduced_potential (state : State) (other : Cell) (digit : Digit) (present : digit ∈ state.candidates other) :
    potential (reduced state other digit) ≤ potential state := by
  have smaller := reduced_weight state other digit present
  simp only [reduced] at smaller
  unfold potential reduced
  simp only [List.length_append]
  split <;> simp only [List.length_singleton, List.length_nil] <;> omega

theorem remove_same_cell (state after : State) (cell other : Cell) (digit : Digit)
    (different : cell ≠ other) (done : remove state other digit = some after) :
    after.candidates cell = state.candidates cell := by
  rcases (remove_some state after other digit).mp done with ⟨_, rfl⟩ | ⟨_, _, rfl⟩
  · rfl
  · simp [reduced, Function.update_of_ne different]

structure Progress (before after : State) : Prop where
  subset : ∀ cell, after.candidates cell ⊆ before.candidates cell
  potential : Propagation.potential after ≤ Propagation.potential before
  weight : Sudoku.weight after.candidates ≤ Sudoku.weight before.candidates
  changed : after.changed = true → before.changed = true ∨ Sudoku.weight after.candidates < Sudoku.weight before.candidates
  retainsChanged : before.changed = true → after.changed = true

theorem Progress.refl (state : State) : Progress state state :=
  ⟨fun _ => Finset.Subset.refl _, Nat.le_refl _, Nat.le_refl _, fun changed => Or.inl changed, id⟩

theorem Progress.trans {a b c : State} (first : Progress a b) (second : Progress b c) : Progress a c := by
  refine ⟨fun cell => (second.subset cell).trans (first.subset cell),
    second.potential.trans first.potential, second.weight.trans first.weight, ?_,
    fun changed => second.retainsChanged (first.retainsChanged changed)⟩
  intro changed
  rcases second.changed changed with before | smaller
  · rcases first.changed before with original | smaller
    · exact Or.inl original
    · exact Or.inr (second.weight.trans_lt smaller)
  · exact Or.inr (smaller.trans_le first.weight)

theorem remove_progress (state after : State) (other : Cell) (digit : Digit)
    (done : remove state other digit = some after) : Progress state after := by
  rcases (remove_some state after other digit).mp done with ⟨_, rfl⟩ | ⟨present, _, rfl⟩
  · exact .refl _
  · exact ⟨reduced_subset state other digit, reduced_potential state other digit present,
      (reduced_weight state other digit present).le, fun _ => Or.inr (reduced_weight state other digit present), fun _ => rfl⟩

theorem remove_preserves (state after : State) (other : Cell) (digit : Digit)
    (done : remove state other digit = some after) (nonempty : Nonempty state) (singles : Singles state) :
    Nonempty after ∧ Singles after := by
  rcases (remove_some state after other digit).mp done with ⟨_, rfl⟩ | ⟨_, remaining, rfl⟩
  · exact ⟨nonempty, singles⟩
  · exact ⟨reduced_nonempty state other digit nonempty remaining, reduced_singles state other digit singles remaining⟩

theorem remove_equiv (state after : State) (cell other : Cell) (digit : Digit)
    (single : state.candidates cell = {digit}) (peer : Peer cell other)
    (done : remove state other digit = some after) (assignment : Assignment) (legal : Legal assignment) :
    Admits after.candidates assignment ↔ Admits state.candidates assignment := by
  rcases (remove_some state after other digit).mp done with ⟨_, rfl⟩ | ⟨_, _, rfl⟩
  · rfl
  · exact erase_peer_iff state.candidates assignment legal single peer

theorem remove_impossible (state : State) (cell other : Cell) (digit : Digit)
    (single : state.candidates cell = {digit}) (peer : Peer cell other) (done : remove state other digit = none) :
    ¬ ∃ assignment, Legal assignment ∧ Admits state.candidates assignment := by
  have empty := ((remove_none state other digit).mp done).2
  rintro ⟨assignment, legal, admitted⟩
  exact empty_cell_impossible (Function.update state.candidates other ((state.candidates other).erase digit))
    (cell := other) (by simpa using empty)
    ⟨assignment, (erase_peer_iff state.candidates assignment legal single peer).mpr admitted⟩

theorem peers_some (state after : State) (cell : Cell) (digit : Digit) (others : List Cell)
    (single : state.candidates cell = {digit}) (related : ∀ other ∈ others, Peer cell other)
    (nonempty : Nonempty state) (singles : Singles state) (done : peers state digit others = some after) :
    Progress state after ∧ Nonempty after ∧ Singles after ∧ after.candidates cell = {digit} ∧
      (∀ assignment, Legal assignment → (Admits after.candidates assignment ↔ Admits state.candidates assignment)) := by
  induction others generalizing state with
  | nil =>
      have same : state = after := Option.some.inj done
      subst state
      exact ⟨.refl _, nonempty, singles, single, fun _ _ => Iff.rfl⟩
  | cons other rest ih =>
      cases step : remove state other digit with
      | none => simp [peers, step] at done
      | some middle =>
          have peer := related other (by simp)
          have kept := remove_preserves state middle other digit step nonempty singles
          have atCell := (remove_same_cell state middle cell other digit peer.1 step).trans single
          have tail : peers middle digit rest = some after := by simpa [peers, step] using done
          obtain ⟨progress, nonempty, singles, single, equivalent⟩ := ih middle atCell
            (fun position member => related position (by simp [member])) kept.1 kept.2 tail
          exact ⟨(remove_progress state middle other digit step).trans progress, nonempty, singles, single,
            fun assignment legal => (equivalent assignment legal).trans
              (remove_equiv state middle cell other digit ‹state.candidates cell = {digit}› peer step assignment legal)⟩

theorem peers_impossible (state : State) (cell : Cell) (digit : Digit) (others : List Cell)
    (single : state.candidates cell = {digit}) (related : ∀ other ∈ others, Peer cell other)
    (done : peers state digit others = none) :
    ¬ ∃ assignment, Legal assignment ∧ Admits state.candidates assignment := by
  induction others generalizing state with
  | nil => simp [peers] at done
  | cons other rest ih =>
      have peer := related other (by simp)
      cases step : remove state other digit with
      | none => exact remove_impossible state cell other digit single peer step
      | some middle =>
          have atCell := (remove_same_cell state middle cell other digit peer.1 step).trans single
          have tail : peers middle digit rest = none := by simpa [peers, step] using done
          have impossible := ih middle atCell (fun position member => related position (by simp [member])) tail
          rintro ⟨assignment, legal, admitted⟩
          exact impossible ⟨assignment, legal, (remove_equiv state middle cell other digit single peer step assignment legal).mpr admitted⟩

end
end ProjectEulerSolutions.Sudoku.Propagation
