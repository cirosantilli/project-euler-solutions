import ProjectEulerSolutions.Lib.IntegerEliminationRow
import Mathlib.Data.Int.GCD
import Mathlib.Data.List.OfFn

/-! Exact greatest-common-divisor normalization for integer Gaussian elimination. -/
namespace ProjectEulerSolutions.IntegerGcdRow
open scoped BigOperators
set_option Elab.async false

theorem fold_dvd_initial (values : List Nat) (initial : Nat) : values.foldl Nat.gcd initial ∣ initial := by
  induction values generalizing initial with
  | nil => exact dvd_refl initial
  | cons first rest ih => exact (ih (Nat.gcd initial first)).trans (Nat.gcd_dvd_left initial first)

theorem fold_dvd_member (values : List Nat) (initial value : Nat) (member : value ∈ values) :
    values.foldl Nat.gcd initial ∣ value := by
  induction values generalizing initial with
  | nil => cases member
  | cons first rest ih =>
      rcases List.mem_cons.mp member with rfl | later
      · exact (fold_dvd_initial rest (Nat.gcd initial value)).trans (Nat.gcd_dvd_right initial value)
      · exact ih _ later

/-- The signed Python accumulator is always the nonnegative gcd of the absolute values seen so far. -/
theorem fold_integers (values : List Int) (initial : Nat) :
    values.foldl (fun divisor value => (Int.gcd divisor value : Int)) initial =
      (((values.map Int.natAbs).foldl Nat.gcd initial : Nat) : Int) := by
  induction values generalizing initial with
  | nil => rfl
  | cons first rest ih =>
      simpa only [List.foldl_cons, List.map_cons, Int.gcd_def, Int.natAbs_natCast] using
        ih (Nat.gcd initial first.natAbs)

def divisor {n : Nat} (row : Fin n → Int) : Nat :=
  (List.ofFn fun column => (row column).natAbs).foldl Nat.gcd 0

theorem divisor_dvd {n : Nat} (row : Fin n → Int) (column : Fin n) : (divisor row : Int) ∣ row column := by
  have divides := fold_dvd_member (List.ofFn fun column => (row column).natAbs) 0 (row column).natAbs (by simp)
  exact Int.natCast_dvd.mpr divides

theorem divisor_nonzero {n : Nat} (row : Fin n → Int) (column : Fin n) (nonzero : row column ≠ 0) :
    divisor row ≠ 0 := by
  intro zero
  have divides := divisor_dvd row column
  rw [zero, Nat.cast_zero, zero_dvd_iff] at divides
  exact nonzero divides

def normalize {n : Nat} (row : Fin n → Int) : Fin n → Int :=
  if divisor row = 0 then row else fun column => row column / (divisor row : Int)

theorem source_fold {n : Nat} (row : Fin n → Int) :
    (List.ofFn row).foldl (fun divisor value => (Int.gcd divisor value : Int)) 0 = (divisor row : Int) := by
  simpa only [List.map_ofFn, Function.comp_def] using fold_integers (List.ofFn row) 0

theorem source_division {n : Nat} (row : Fin n → Int) (nonzero : divisor row ≠ 0) (column : Fin n) :
    (row column).fdiv (divisor row) = normalize row column := by
  rw [Int.fdiv_eq_ediv_of_dvd (divisor_dvd row column)]
  simp [normalize, nonzero]

/-- Reducing a row by its gcd preserves every rational augmented equation, including a zero row. -/
theorem equation_iff {n : Nat} (row : Fin n → Int) (value : Fin n → ℚ) :
    (∑ column, (normalize row column : ℚ) * value column) = 0 ↔
      (∑ column, (row column : ℚ) * value column) = 0 := by
  by_cases zero : divisor row = 0
  · simp [normalize, zero]
  · have nonzero : (divisor row : Int) ≠ 0 := by exact_mod_cast zero
    have same := IntegerEliminationRow.equation_iff 1 0 (divisor row) (fun _ : Fin n => 0) row value
      (by decide) nonzero (fun column => by simpa using divisor_dvd row column) (by simp)
    simpa [normalize, zero, IntegerEliminationRow.reduce] using same

theorem normalize_zero {n : Nat} (row : Fin n → Int) (column : Fin n) (zero : row column = 0) :
    normalize row column = 0 := by
  unfold normalize
  split <;> simp [zero]

end ProjectEulerSolutions.IntegerGcdRow
