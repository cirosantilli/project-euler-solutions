import Mathlib.Data.Finset.Lattice.Fold
import Mathlib.Data.Int.Basic
import Mathlib.Data.Nat.Find
import Mathlib.Tactic

/-! The initial run of positive integers depends only on membership in a finite list. -/
namespace ProjectEulerSolutions.PositiveRun
set_option Elab.async false

/-- The successor of the largest absolute value is absent from any finite input list. -/
theorem exists_missing (xs : List Int) : ∃ n : Nat, 0 < n ∧ (n : Int) ∉ xs := by
  let bound := xs.toFinset.sup Int.natAbs
  refine ⟨bound + 1, by omega, ?_⟩
  intro member
  have within : Int.natAbs ((bound + 1 : Nat) : Int) ≤ bound :=
    Finset.le_sup (List.mem_toFinset.mpr member)
  simp only [Int.natAbs_natCast] at within
  omega

def firstMissing (xs : List Int) : Nat := Nat.find (exists_missing xs)

theorem positive (xs : List Int) : 0 < firstMissing xs := (Nat.find_spec (exists_missing xs)).1

theorem missing (xs : List Int) : (firstMissing xs : Int) ∉ xs := (Nat.find_spec (exists_missing xs)).2

theorem before (xs : List Int) (n : Nat) (positiveN : 0 < n) (earlier : n < firstMissing xs) :
    (n : Int) ∈ xs := by
  by_contra absent
  have minimal := Nat.find_min' (exists_missing xs) ⟨positiveN, absent⟩
  change firstMissing xs ≤ n at minimal
  omega

def length (xs : List Int) : Nat := firstMissing xs - 1

theorem firstMissing_eq_length (xs : List Int) : firstMissing xs = length xs + 1 := by
  have := positive xs
  unfold length
  omega

/-- Duplicates, negative values, and ordering cannot change the positive run. -/
theorem congr (xs ys : List Int)
    (members : ∀ n : Nat, 0 < n → ((n : Int) ∈ xs ↔ (n : Int) ∈ ys)) : length xs = length ys := by
  have left : firstMissing xs ≤ firstMissing ys :=
    Nat.find_min' (exists_missing xs) ⟨positive ys, fun member =>
      missing ys ((members _ (positive ys)).mp member)⟩
  have right : firstMissing ys ≤ firstMissing xs :=
    Nat.find_min' (exists_missing ys) ⟨positive xs, fun member =>
      missing xs ((members _ (positive xs)).mpr member)⟩
  unfold length
  congr 1
  omega

end ProjectEulerSolutions.PositiveRun
