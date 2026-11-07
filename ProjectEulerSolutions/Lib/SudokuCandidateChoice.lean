import ProjectEulerSolutions.Lib.SudokuCandidateArray

/-! Minimum remaining values, with Python's row-major index tie break. -/
namespace ProjectEulerSolutions.Sudoku.Choice
set_option Elab.async false
noncomputable section

def count (candidates : Candidates) (slot : Fin 81) : Nat := (candidates (Encoding.cell slot)).card
def priority (candidates : Candidates) (slot : Fin 81) : Nat := 81 * count candidates slot + slot.val

theorem priority_lt (candidates : Candidates) (left right : Fin 81) :
    priority candidates left < priority candidates right ↔
      count candidates left < count candidates right ∨
        count candidates left = count candidates right ∧ left.val < right.val := by
  unfold priority
  have := left.isLt
  have := right.isLt
  omega

def consider (candidates : Candidates) (best : Option (Fin 81)) (slot : Fin 81) : Option (Fin 81) :=
  if 1 < count candidates slot then
    match best with
    | none => some slot
    | some old => some (if priority candidates slot < priority candidates old then slot else old)
  else best

def Best (candidates : Candidates) (seen : List (Fin 81)) : Option (Fin 81) → Prop
  | none => ∀ slot ∈ seen, ¬ 1 < count candidates slot
  | some best => best ∈ seen ∧ 1 < count candidates best ∧
      ∀ slot ∈ seen, 1 < count candidates slot → priority candidates best ≤ priority candidates slot

theorem initial (candidates : Candidates) : Best candidates [] none := by simp [Best]

theorem step (candidates : Candidates) (seen : List (Fin 81)) (best : Option (Fin 81)) (slot : Fin 81)
    (known : Best candidates seen best) : Best candidates (seen ++ [slot]) (consider candidates best slot) := by
  by_cases eligible : 1 < count candidates slot
  · cases best with
    | none =>
        simp only [consider, if_pos eligible, Best]
        refine ⟨by simp, eligible, ?_⟩
        intro other member multiple
        rcases List.mem_append.mp member with old | current
        · exact False.elim (known other old multiple)
        · simpa only [List.mem_singleton.mp current] using Nat.le_refl (priority candidates slot)
    | some old =>
        rcases known with ⟨member, multiple, minimal⟩
        by_cases better : priority candidates slot < priority candidates old
        · simp only [consider, if_pos eligible, if_pos better, Best]
          refine ⟨by simp, eligible, ?_⟩
          intro other selected otherMultiple
          rcases List.mem_append.mp selected with earlier | current
          · exact better.le.trans (minimal other earlier otherMultiple)
          · simpa only [List.mem_singleton.mp current] using Nat.le_refl (priority candidates slot)
        · simp only [consider, if_pos eligible, if_neg better, Best]
          refine ⟨List.mem_append_left _ member, multiple, ?_⟩
          intro other selected otherMultiple
          rcases List.mem_append.mp selected with earlier | current
          · exact minimal other earlier otherMultiple
          · simpa only [List.mem_singleton.mp current] using Nat.le_of_not_gt better
  · simp only [consider, if_neg eligible]
    cases best with
    | none =>
        intro other member
        rcases List.mem_append.mp member with earlier | current
        · exact known other earlier
        · simpa only [List.mem_singleton.mp current] using eligible
    | some old =>
        rcases known with ⟨member, multiple, minimal⟩
        refine ⟨List.mem_append_left _ member, multiple, ?_⟩
        intro other selected otherMultiple
        rcases List.mem_append.mp selected with earlier | current
        · exact minimal other earlier otherMultiple
        · exact False.elim (eligible (List.mem_singleton.mp current ▸ otherMultiple))

theorem available (candidates : Candidates) (nonempty : ∀ cell, candidates cell ≠ ∅)
    (unsolved : ¬ ∀ cell, (candidates cell).card = 1) : ∃ slot : Fin 81, 1 < count candidates slot := by
  by_contra absent
  apply unsolved
  intro cell
  have positive := Finset.card_pos.mpr (Finset.nonempty_iff_ne_empty.mpr (nonempty cell))
  have upper : ¬ 1 < count candidates (Encoding.index cell) := fun multiple => absent ⟨_, multiple⟩
  simp only [count, Encoding.cell_index] at upper
  omega

theorem finish (candidates : Candidates) (result : Option (Fin 81))
    (known : Best candidates (List.finRange 81) result)
    (nonempty : ∀ cell, candidates cell ≠ ∅) (unsolved : ¬ ∀ cell, (candidates cell).card = 1) :
    ∃ slot, result = some slot ∧ 1 < count candidates slot ∧
      ∀ other, 1 < count candidates other → priority candidates slot ≤ priority candidates other := by
  cases result with
  | none =>
      obtain ⟨slot, multiple⟩ := available candidates nonempty unsolved
      exact False.elim (known slot (by simp) multiple)
  | some slot => exact ⟨slot, rfl, known.2.1, fun other multiple => known.2.2 other (by simp) multiple⟩

end
end ProjectEulerSolutions.Sudoku.Choice
