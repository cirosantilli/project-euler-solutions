import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Data.Rat.Cast.Order

/-! Exact integer row reduction preserves the rational solution space. -/
namespace ProjectEulerSolutions.IntegerEliminationRow
open scoped BigOperators
set_option Elab.async false

variable {ι : Type*}

def reduce (pivot factor previous : Int) (pivotRow row : ι → Int) : ι → Int :=
  fun column => (pivot * row column - factor * pivotRow column) / previous

theorem scaled_entry (pivot factor previous : Int) (pivotRow row : ι → Int) (column : ι)
    (exactDivision : previous ∣ pivot * row column - factor * pivotRow column) :
    (previous : ℚ) * reduce pivot factor previous pivotRow row column =
      (pivot : ℚ) * row column - (factor : ℚ) * pivotRow column := by
  exact_mod_cast Int.mul_ediv_cancel_of_dvd exactDivision

variable [Fintype ι]

theorem scaled_equation (pivot factor previous : Int) (pivotRow row : ι → Int) (value : ι → ℚ)
    (exactDivision : ∀ column, previous ∣ pivot * row column - factor * pivotRow column) :
    (previous : ℚ) * (∑ column, (reduce pivot factor previous pivotRow row column : ℚ) * value column) =
      (pivot : ℚ) * (∑ column, (row column : ℚ) * value column) -
        (factor : ℚ) * (∑ column, (pivotRow column : ℚ) * value column) := by
  simp only [Finset.mul_sum, ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro column _
  rw [← mul_assoc, scaled_entry pivot factor previous pivotRow row column (exactDivision column)]
  simp only [sub_mul, mul_assoc]

/-- Given the pivot equation, exact division by a nonzero previous pivot preserves this row equation. -/
theorem equation_iff (pivot factor previous : Int) (pivotRow row : ι → Int) (value : ι → ℚ)
    (pivotNonzero : pivot ≠ 0) (previousNonzero : previous ≠ 0)
    (exactDivision : ∀ column, previous ∣ pivot * row column - factor * pivotRow column)
    (pivotEquation : ∑ column, (pivotRow column : ℚ) * value column = 0) :
    (∑ column, (reduce pivot factor previous pivotRow row column : ℚ) * value column) = 0 ↔
      (∑ column, (row column : ℚ) * value column) = 0 := by
  have scaled := scaled_equation pivot factor previous pivotRow row value exactDivision
  rw [pivotEquation, mul_zero, sub_zero] at scaled
  have pivotNotZero : (pivot : ℚ) ≠ 0 := by exact_mod_cast pivotNonzero
  have previousNotZero : (previous : ℚ) ≠ 0 := by exact_mod_cast previousNonzero
  constructor
  · intro zero
    rw [zero, mul_zero] at scaled
    exact (mul_eq_zero.mp scaled.symm).resolve_left pivotNotZero
  · intro zero
    rw [zero, mul_zero] at scaled
    exact (mul_eq_zero.mp scaled).resolve_left previousNotZero

end ProjectEulerSolutions.IntegerEliminationRow
