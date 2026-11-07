import ProjectEulerSolutions.Lib.IntegerGcdRow
import ProjectEulerSolutions.Lib.IntegerEliminationMatrix

/-! The reference Gaussian step reduces each changed row by its own exact gcd. -/
namespace ProjectEulerSolutions.IntegerGcdElimination
open scoped BigOperators
set_option Elab.async false

variable {n m : Nat}

def row (pivot factor : Int) (pivotRow current : Fin m → Int) : Fin m → Int :=
  if factor = 0 then current else IntegerGcdRow.normalize (fun column => pivot * current column - factor * pivotRow column)

/-- The zero-factor branch keeps the row; every other branch makes an invertible rational row operation. -/
theorem row_equation_iff (pivot factor : Int) (pivotRow current : Fin m → Int) (value : Fin m → ℚ)
    (nonzero : pivot ≠ 0) (pivotEquation : ∑ column, (pivotRow column : ℚ) * value column = 0) :
    (∑ column, (row pivot factor pivotRow current column : ℚ) * value column) = 0 ↔
      (∑ column, (current column : ℚ) * value column) = 0 := by
  by_cases zero : factor = 0
  · simp [row, zero]
  · rw [row, if_neg zero, IntegerGcdRow.equation_iff]
    simpa only [IntegerEliminationRow.reduce, Int.ediv_one] using
      IntegerEliminationRow.equation_iff pivot factor 1 pivotRow current value nonzero (by decide)
        (by intro column; exact one_dvd _) pivotEquation

def reduce (matrix : Matrix (Fin n) (Fin m) Int) (pivotRow : Fin n) (pivotColumn : Fin m) : Matrix (Fin n) (Fin m) Int :=
  fun index => if index.val ≤ pivotRow.val then matrix index else
    row (matrix pivotRow pivotColumn) (matrix index pivotColumn) (matrix pivotRow) (matrix index)

theorem equations_iff (matrix : Matrix (Fin n) (Fin m) Int) (pivotRow : Fin n) (pivotColumn : Fin m)
    (nonzero : matrix pivotRow pivotColumn ≠ 0) (value : Fin m → ℚ) :
    (∀ index, ∑ column, (reduce matrix pivotRow pivotColumn index column : ℚ) * value column = 0) ↔
      (∀ index, ∑ column, (matrix index column : ℚ) * value column = 0) := by
  constructor
  · intro reduced
    have pivotEquation : ∑ column, (matrix pivotRow column : ℚ) * value column = 0 := by
      simpa [reduce] using reduced pivotRow
    intro index
    by_cases kept : index.val ≤ pivotRow.val
    · simpa [reduce, kept] using reduced index
    · apply (row_equation_iff _ _ (matrix pivotRow) (matrix index) value nonzero pivotEquation).mp
      simpa only [reduce, if_neg kept] using reduced index
  · intro original index
    by_cases kept : index.val ≤ pivotRow.val
    · simpa only [reduce, if_pos kept] using original index
    · have transformed := (row_equation_iff _ (matrix index pivotColumn) (matrix pivotRow) (matrix index) value nonzero (original pivotRow)).mpr (original index)
      simpa only [reduce, if_neg kept] using transformed

/-- Normalizing augmented rows retains injectivity of their coefficient matrix. -/
theorem injective (matrix : Matrix (Fin n) (Fin (n + 1)) Int) (pivot : Fin n)
    (original : Function.Injective (Matrix.mulVec (fun index column : Fin n => (matrix index column.castSucc : ℚ))))
    (nonzero : matrix pivot pivot.castSucc ≠ 0) :
    Function.Injective (Matrix.mulVec (fun index column : Fin n => (reduce matrix pivot pivot.castSucc index column.castSucc : ℚ))) := by
  intro left right equal
  have reducedZero : Matrix.mulVec
      (fun index column : Fin n => (reduce matrix pivot pivot.castSucc index column.castSucc : ℚ)) (left - right) = 0 := by
    rw [Matrix.mulVec_sub, equal, sub_self]
  have originalZero := (equations_iff matrix pivot pivot.castSucc nonzero
    (Fin.snoc (α := fun _ => ℚ) (left - right) 0)).mp (by
      intro index
      simpa only [Fin.sum_univ_castSucc, Fin.snoc_castSucc, Fin.snoc_last, mul_zero, add_zero] using congrFun reducedZero index)
  apply original
  funext index
  have equation := originalZero index
  simpa only [Fin.sum_univ_castSucc, Fin.snoc_castSucc, Fin.snoc_last, mul_zero, add_zero,
    Pi.sub_apply, mul_sub, Finset.sum_sub_distrib, sub_eq_zero] using equation

/-- Gcd division keeps existing zeros and the newly eliminated pivot column equal to zero. -/
theorem zeros (matrix : Matrix (Fin n) (Fin (n + 1)) Int) (pivot : Fin n)
    (eliminated : ∀ index column, column.val < pivot.val → column.val < index.val → matrix index column = 0)
    (index : Fin n) (column : Fin (n + 1)) (processed : column.val ≤ pivot.val) (below : column.val < index.val) :
    reduce matrix pivot pivot.castSucc index column = 0 := by
  by_cases kept : index.val ≤ pivot.val
  · simpa only [reduce, if_pos kept] using eliminated index column (by omega) below
  · rw [reduce, if_neg kept, row]
    split_ifs with zero
    · by_cases earlier : column.val < pivot.val
      · exact eliminated index column earlier below
      · have same : column = pivot.castSucc := Fin.ext (by simp; omega)
        simpa only [same] using zero
    · apply IntegerGcdRow.normalize_zero
      by_cases earlier : column.val < pivot.val
      · rw [eliminated index column earlier below, eliminated pivot column earlier earlier]
        simp
      · have same : column = pivot.castSucc := Fin.ext (by simp; omega)
        subst column
        simp [mul_comm]

end ProjectEulerSolutions.IntegerGcdElimination
