import ProjectEulerSolutions.Lib.IntegerEliminationRow
import Mathlib.LinearAlgebra.Matrix.ToLin

/-! A complete fraction-free column reduction preserves every augmented solution. -/
namespace ProjectEulerSolutions.IntegerEliminationMatrix
open scoped BigOperators
set_option Elab.async false

variable {n : Nat} {ι : Type*} [Fintype ι]

def reduce (matrix : Matrix (Fin n) ι Int) (pivotRow : Fin n) (pivotColumn : ι) (previous : Int) : Matrix (Fin n) ι Int :=
  fun row column => if row.val ≤ pivotRow.val then matrix row column else
    IntegerEliminationRow.reduce (matrix pivotRow pivotColumn) (matrix row pivotColumn) previous (matrix pivotRow) (matrix row) column

theorem equations_iff (matrix : Matrix (Fin n) ι Int) (pivotRow : Fin n) (pivotColumn : ι) (previous : Int)
    (pivotNonzero : matrix pivotRow pivotColumn ≠ 0) (previousNonzero : previous ≠ 0)
    (exactDivision : ∀ row : Fin n, pivotRow.val < row.val → ∀ column,
      previous ∣ matrix pivotRow pivotColumn * matrix row column - matrix row pivotColumn * matrix pivotRow column)
    (value : ι → ℚ) :
    (∀ row, ∑ column, (reduce matrix pivotRow pivotColumn previous row column : ℚ) * value column = 0) ↔
      (∀ row, ∑ column, (matrix row column : ℚ) * value column = 0) := by
  constructor
  · intro reduced
    have pivotEquation : ∑ column, (matrix pivotRow column : ℚ) * value column = 0 := by
      simpa [reduce] using reduced pivotRow
    intro row
    by_cases kept : row.val ≤ pivotRow.val
    · simpa [reduce, kept] using reduced row
    · apply (IntegerEliminationRow.equation_iff _ _ previous (matrix pivotRow) (matrix row) value
        pivotNonzero previousNonzero (exactDivision row (by omega)) pivotEquation).mp
      simpa [reduce, kept] using reduced row
  · intro original row
    by_cases kept : row.val ≤ pivotRow.val
    · simpa [reduce, kept] using original row
    · have reduced := (IntegerEliminationRow.equation_iff _ _ previous (matrix pivotRow) (matrix row) value
        pivotNonzero previousNonzero (exactDivision row (by omega)) (original pivotRow)).mpr (original row)
      simpa [reduce, kept] using reduced

/-- Exact column reduction keeps the coefficient matrix injective, allowing the next pivot search. -/
theorem injective (matrix : Matrix (Fin n) (Fin n) Int) (pivot : Fin n) (previous : Int)
    (original : Function.Injective (Matrix.mulVec (fun row column => (matrix row column : ℚ))))
    (pivotNonzero : matrix pivot pivot ≠ 0) (previousNonzero : previous ≠ 0)
    (exactDivision : ∀ row : Fin n, pivot.val < row.val → ∀ column,
      previous ∣ matrix pivot pivot * matrix row column - matrix row pivot * matrix pivot column) :
    Function.Injective (Matrix.mulVec (fun row column => (reduce matrix pivot pivot previous row column : ℚ))) := by
  intro left right equal
  have reducedZero : Matrix.mulVec (fun row column => (reduce matrix pivot pivot previous row column : ℚ)) (left - right) = 0 := by
    rw [Matrix.mulVec_sub, equal, sub_self]
  have originalZero : Matrix.mulVec (fun row column => (matrix row column : ℚ)) (left - right) = 0 := by
    funext row
    exact (equations_iff matrix pivot pivot previous pivotNonzero previousNonzero exactDivision (left - right)).mp
      (fun row => congrFun reducedZero row) row
  apply original
  rw [Matrix.mulVec_sub] at originalZero
  exact sub_eq_zero.mp originalZero

/-- The eliminated prefix extends by one column, including the pivot entry that Python writes as zero. -/
theorem zeros (matrix : Matrix (Fin n) (Fin n) Int) (pivot : Fin n) (previous : Int)
    (eliminated : ∀ row column : Fin n, column.val < pivot.val → column.val < row.val → matrix row column = 0)
    (row column : Fin n) (processed : column.val ≤ pivot.val) (below : column.val < row.val) :
    reduce matrix pivot pivot previous row column = 0 := by
  by_cases kept : row.val ≤ pivot.val
  · simp only [reduce, if_pos kept]
    exact eliminated row column (by omega) below
  · simp only [reduce, if_neg kept, IntegerEliminationRow.reduce]
    by_cases earlier : column.val < pivot.val
    · rw [eliminated row column earlier below, eliminated pivot column earlier earlier]
      simp
    · have same : column = pivot := Fin.ext (by omega)
      subst column
      simp [mul_comm]

end ProjectEulerSolutions.IntegerEliminationMatrix
