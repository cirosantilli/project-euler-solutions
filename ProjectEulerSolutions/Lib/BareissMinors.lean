import ProjectEulerSolutions.Lib.BorderedDeterminants
import Mathlib.Data.Fin.Tuple.Basic
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Tactic.FinCases

/-! Ordered minors supply the integer values in each active Bareiss submatrix. -/
namespace ProjectEulerSolutions.BareissMinors
set_option Elab.async false

variable {α β : Type*} {k : Nat}

def minor (matrix : Matrix α β Int) (rows : Fin k → α) (columns : Fin k → β) : Int :=
  (matrix.submatrix rows columns).det

private theorem snoc_sum (values : Fin k → α) (last : α) (index : Fin k ⊕ Fin 1) :
    Fin.snoc (α := fun _ => α) values last (finSumFinEquiv (m := k) (n := 1) index) = Sum.elim values (fun _ => last) index := by
  cases index with
  | inl index => exact Fin.snoc_castSucc (α := fun _ => α) last values index
  | inr index =>
      have only : index = 0 := Subsingleton.elim _ _
      subst index
      exact Fin.snoc_last (α := fun _ => α) last values

private theorem snoc_two_sum (values : Fin k → α) (first last : α) (index : Fin k ⊕ Fin 2) :
    Fin.snoc (α := fun _ => α) (Fin.snoc (α := fun _ => α) values first) last
      (finSumFinEquiv (m := k) (n := 2) index) = Sum.elim values ![first, last] index := by
  cases index with
  | inl index =>
      change Fin.snoc (α := fun _ => α) (Fin.snoc (α := fun _ => α) values first) last index.castSucc.castSucc = values index
      rw [Fin.snoc_castSucc, Fin.snoc_castSucc]
  | inr index =>
      fin_cases index
      · change Fin.snoc (α := fun _ => α) (Fin.snoc (α := fun _ => α) values first) last (Fin.last k).castSucc = first
        rw [Fin.snoc_castSucc, Fin.snoc_last]
      · exact Fin.snoc_last (α := fun _ => α) last (Fin.snoc (α := fun _ => α) values first)

theorem snoc_minor (matrix : Matrix α β Int) (prefixRows : Fin k → α) (prefixColumns : Fin k → β)
    (rows : Fin 2 → α) (columns : Fin 2 → β) (row column : Fin 2) :
    minor matrix (Fin.snoc prefixRows (rows row)) (Fin.snoc prefixColumns (columns column)) =
      (BorderedDeterminants.border (matrix.submatrix prefixRows prefixColumns) (matrix.submatrix prefixRows columns)
        (matrix.submatrix rows prefixColumns) (matrix.submatrix rows columns) row column).det := by
  unfold minor
  rw [← Matrix.det_submatrix_equiv_self (finSumFinEquiv (m := k) (n := 1))]
  congr 1
  ext left right
  simp only [Matrix.submatrix_apply, snoc_sum]
  cases left <;> cases right <;> rfl

theorem snoc_two_minor (matrix : Matrix α β Int) (prefixRows : Fin k → α) (prefixColumns : Fin k → β)
    (firstRow lastRow : α) (firstColumn lastColumn : β) :
    minor matrix (Fin.snoc (Fin.snoc prefixRows firstRow) lastRow) (Fin.snoc (Fin.snoc prefixColumns firstColumn) lastColumn) =
      (Matrix.fromBlocks (matrix.submatrix prefixRows prefixColumns) (matrix.submatrix prefixRows ![firstColumn, lastColumn])
        (matrix.submatrix ![firstRow, lastRow] prefixColumns) (matrix.submatrix ![firstRow, lastRow] ![firstColumn, lastColumn])).det := by
  unfold minor
  rw [← Matrix.det_submatrix_equiv_self (finSumFinEquiv (m := k) (n := 2))]
  congr 1
  ext left right
  simp only [Matrix.submatrix_apply, snoc_two_sum]
  cases left <;> cases right <;> rfl

/-- A Bareiss numerator is exactly the previous minor times the next minor. -/
theorem identity (matrix : Matrix α β Int) (prefixRows : Fin k → α) (prefixColumns : Fin k → β)
    (nonzero : minor matrix prefixRows prefixColumns ≠ 0) (pivotRow row : α) (pivotColumn column : β) :
    minor matrix prefixRows prefixColumns *
        minor matrix (Fin.snoc (Fin.snoc prefixRows pivotRow) row) (Fin.snoc (Fin.snoc prefixColumns pivotColumn) column) =
      minor matrix (Fin.snoc prefixRows pivotRow) (Fin.snoc prefixColumns pivotColumn) *
        minor matrix (Fin.snoc prefixRows row) (Fin.snoc prefixColumns column) -
      minor matrix (Fin.snoc prefixRows row) (Fin.snoc prefixColumns pivotColumn) *
        minor matrix (Fin.snoc prefixRows pivotRow) (Fin.snoc prefixColumns column) := by
  rw [snoc_two_minor]
  simpa only [minor, ← snoc_minor, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_fin_one] using
    BorderedDeterminants.int_identity (matrix.submatrix prefixRows prefixColumns) nonzero
      (matrix.submatrix prefixRows ![pivotColumn, column]) (matrix.submatrix ![pivotRow, row] prefixColumns)
      (matrix.submatrix ![pivotRow, row] ![pivotColumn, column])

end ProjectEulerSolutions.BareissMinors
