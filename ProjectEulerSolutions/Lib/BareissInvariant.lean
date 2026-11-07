import ProjectEulerSolutions.Lib.BareissMinors
import ProjectEulerSolutions.Lib.IntegerEliminationMatrix

/-! The active submatrix consists of ordered minors of the original integer matrix. -/
namespace ProjectEulerSolutions.BareissInvariant
open BareissMinors
set_option Elab.async false

variable {n m k : Nat}

def Active (original current : Matrix (Fin n) (Fin m) Int)
    (prefixRows : Fin k → Fin n) (prefixColumns : Fin k → Fin m) (labels : Fin n → Fin n) : Prop :=
  ∀ row column, k ≤ row.val → k ≤ column.val →
    current row column = minor original (Fin.snoc prefixRows (labels row)) (Fin.snoc prefixColumns column)

theorem initial (original : Matrix (Fin n) (Fin m) Int) :
    Active original original (Fin.elim0 : Fin 0 → Fin n) (Fin.elim0 : Fin 0 → Fin m) id := by
  intro row column _ _
  simp [minor, Matrix.submatrix_apply, Fin.snoc_zero]

theorem initial_previous (original : Matrix (Fin n) (Fin m) Int) :
    minor original (Fin.elim0 : Fin 0 → Fin n) (Fin.elim0 : Fin 0 → Fin m) = 1 := by
  simp [minor]

theorem Active.reindex {original current : Matrix (Fin n) (Fin m) Int}
    {prefixRows : Fin k → Fin n} {prefixColumns : Fin k → Fin m} {labels : Fin n → Fin n}
    (active : Active original current prefixRows prefixColumns labels) (rename : Fin n → Fin n)
    (preserves : ∀ row, k ≤ row.val → k ≤ (rename row).val) :
    Active original (fun row column => current (rename row) column) prefixRows prefixColumns (labels ∘ rename) :=
  fun row column rowBound columnBound => active (rename row) column (preserves row rowBound) columnBound

/-- Every division in an active row is exact, including its already-zero prefix. -/
theorem Active.exact_division {original current : Matrix (Fin n) (Fin m) Int}
    {prefixRows : Fin k → Fin n} {prefixColumns : Fin k → Fin m} {labels : Fin n → Fin n}
    (active : Active original current prefixRows prefixColumns labels)
    (nonzero : minor original prefixRows prefixColumns ≠ 0)
    (pivot : Fin n) (target : Fin m) (pivotIndex : pivot.val = k) (targetIndex : target.val = k)
    (eliminated : ∀ row column, column.val < k → column.val < row.val → current row column = 0)
    (row : Fin n) (below : pivot.val < row.val) (column : Fin m) :
    minor original prefixRows prefixColumns ∣ current pivot target * current row column - current row target * current pivot column := by
  by_cases earlier : column.val < k
  · rw [eliminated row column earlier (by omega), eliminated pivot column earlier (by omega)]
    simp
  · rw [active pivot target (by omega) (by omega), active row column (by omega) (by omega),
      active row target (by omega) (by omega), active pivot column (by omega) (by omega)]
    exact ⟨minor original (Fin.snoc (Fin.snoc prefixRows (labels pivot)) (labels row))
        (Fin.snoc (Fin.snoc prefixColumns target) column),
      (identity original prefixRows prefixColumns nonzero (labels pivot) (labels row) target column).symm⟩

/-- The quotient is the next ordered minor, so the invariant survives an entire column reduction. -/
theorem Active.advance {original current : Matrix (Fin n) (Fin m) Int}
    {prefixRows : Fin k → Fin n} {prefixColumns : Fin k → Fin m} {labels : Fin n → Fin n}
    (active : Active original current prefixRows prefixColumns labels)
    (nonzero : minor original prefixRows prefixColumns ≠ 0)
    (pivot : Fin n) (target : Fin m) (pivotIndex : pivot.val = k) (targetIndex : target.val = k) :
    Active original (IntegerEliminationMatrix.reduce current pivot target (minor original prefixRows prefixColumns))
      (Fin.snoc prefixRows (labels pivot)) (Fin.snoc prefixColumns target) labels := by
  intro row column rowBound columnBound
  rw [IntegerEliminationMatrix.reduce, if_neg (by omega), IntegerEliminationRow.reduce,
    active pivot target (by omega) (by omega), active row column (by omega) (by omega),
    active row target (by omega) (by omega), active pivot column (by omega) (by omega)]
  rw [← identity original prefixRows prefixColumns nonzero (labels pivot) (labels row) target column]
  exact Int.mul_ediv_cancel_left _ nonzero

theorem Active.next_previous {original current : Matrix (Fin n) (Fin m) Int}
    {prefixRows : Fin k → Fin n} {prefixColumns : Fin k → Fin m} {labels : Fin n → Fin n}
    (active : Active original current prefixRows prefixColumns labels)
    (pivot : Fin n) (target : Fin m) (pivotIndex : pivot.val = k) (targetIndex : target.val = k)
    (pivotNonzero : current pivot target ≠ 0) :
    minor original (Fin.snoc prefixRows (labels pivot)) (Fin.snoc prefixColumns target) ≠ 0 := by
  rw [← active pivot target (by omega) (by omega)]
  exact pivotNonzero

end ProjectEulerSolutions.BareissInvariant
