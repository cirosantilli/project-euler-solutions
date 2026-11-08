import ProjectEulerSolutions.Lib.MonopolyCardCounts
import ProjectEulerSolutions.Lib.MonopolyEncoded

/-! Dense transition entries counted by the reference's explicit card loops. -/
namespace ProjectEulerSolutions.Monopoly
set_option Elab.async false

/-- Incrementing one encoded source/destination entry leaves all other entries unchanged. -/
def bump (counts : Matrix (Fin 120) (Fin 120) Int) (source target : Nat) : Matrix (Fin 120) (Fin 120) Int :=
  fun row column => counts row column + if row.val = source ∧ column.val = target then 1 else 0

theorem bump_update (counts : Matrix (Fin 120) (Fin 120) Int) (source target : Fin 120) :
    bump counts source.val target.val =
      Function.update counts source (Function.update (counts source) target (counts source target + 1)) := by
  funext row column
  by_cases sameRow : row = source
  · subst row
    by_cases sameColumn : column = target
    · subst column
      simp [bump]
    · simp [bump, sameColumn, Fin.val_inj]
  · simp [bump, sameRow, Fin.val_inj]

def cardCount (sides : Nat) (before after : Fin 120) : Int :=
  ((turnCards sides (before.val / 3, before.val % 3)).map fun state =>
    if after.val = encode state.1 state.2 then (1 : Int) else 0).sum

/-- Each explicit card pair contributes one to exactly the optimized encoded transition weight. -/
theorem cardCount_eq (sides : Nat) (before after : Fin 120) : cardCount sides before after = encodedWeight sides before after := by
  rw [cardCount, turnCards_sum]
  unfold encodedWeight
  congr 1
  apply List.map_congr_left
  intro entry _
  split_ifs <;> simp

/-- An arbitrary list of enumerated rolls adds precisely its destination histogram. -/
theorem fold_bump (states : List (Nat × Nat)) (counts : Matrix (Fin 120) (Fin 120) Int)
    (source : Nat) (row column : Fin 120) :
    (states.foldl (fun matrix state => bump matrix source (encode state.1 state.2)) counts) row column =
      counts row column + (states.map fun state =>
        if row.val = source ∧ column.val = encode state.1 state.2 then (1 : Int) else 0).sum := by
  induction states generalizing counts with
  | nil => simp
  | cons first states ih =>
      simp only [List.foldl_cons, ih, bump, List.map_cons, List.sum_cons]
      omega

theorem fold_turnCards (sides : Nat) (source : Fin 120) (counts : Matrix (Fin 120) (Fin 120) Int)
    (row column : Fin 120) :
    ((turnCards sides (source.val / 3, source.val % 3)).foldl
      (fun matrix state => bump matrix source.val (encode state.1 state.2)) counts) row column =
      counts row column + if row = source then encodedWeight sides source column else 0 := by
  rw [fold_bump]
  by_cases same : row = source
  · subst row
    simp only [true_and, if_true]
    exact congrArg (counts source column + ·) (cardCount_eq sides source column)
  · have different : row.val ≠ source.val := fun equal => same (Fin.ext equal)
    simp [different, same]

end ProjectEulerSolutions.Monopoly
