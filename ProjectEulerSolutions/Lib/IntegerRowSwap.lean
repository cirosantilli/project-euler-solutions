import ProjectEulerSolutions.Lib.IntegerEliminationMatrix

/-! Pivot row swaps preserve the equations, rank, and previously eliminated columns. -/
namespace ProjectEulerSolutions.IntegerRowSwap
open scoped BigOperators
set_option Elab.async false

variable {n m k : Nat}

def swap (matrix : Matrix (Fin n) (Fin m) Int) (first second : Fin n) : Matrix (Fin n) (Fin m) Int :=
  fun row column => matrix (Equiv.swap first second row) column

theorem stays_above (first second row : Fin n) (firstBound : k ≤ first.val) (secondBound : k ≤ second.val)
    (rowBound : k ≤ row.val) : k ≤ (Equiv.swap first second row).val := by
  rw [Equiv.swap_apply_def]
  split_ifs <;> assumption

theorem equations_iff (matrix : Matrix (Fin n) (Fin m) Int) (first second : Fin n) (value : Fin m → ℚ) :
    (∀ row, ∑ column, (swap matrix first second row column : ℚ) * value column = 0) ↔
      (∀ row, ∑ column, (matrix row column : ℚ) * value column = 0) := by
  constructor
  · intro swapped row
    simpa only [swap, Equiv.swap_apply_self] using swapped (Equiv.swap first second row)
  · intro original row
    exact original (Equiv.swap first second row)

theorem injective (matrix : Matrix (Fin n) (Fin n) Int) (first second : Fin n)
    (original : Function.Injective (Matrix.mulVec (fun row column => (matrix row column : ℚ)))) :
    Function.Injective (Matrix.mulVec (fun row column => (swap matrix first second row column : ℚ))) := by
  intro left right equal
  apply original
  funext row
  simpa only [Matrix.mulVec, dotProduct, swap, Equiv.swap_apply_self] using congrFun equal (Equiv.swap first second row)

theorem zeros (matrix : Matrix (Fin n) (Fin m) Int) (first second : Fin n)
    (firstBound : k ≤ first.val) (secondBound : k ≤ second.val)
    (eliminated : ∀ row column, column.val < k → column.val < row.val → matrix row column = 0)
    (row : Fin n) (column : Fin m) (earlier : column.val < k) (below : column.val < row.val) :
    swap matrix first second row column = 0 := by
  apply eliminated _ column earlier
  by_cases active : k ≤ row.val
  · have bound := stays_above first second row firstBound secondBound active
    omega
  · rw [Equiv.swap_apply_of_ne_of_ne (by intro same; subst row; omega) (by intro same; subst row; omega)]
    exact below

end ProjectEulerSolutions.IntegerRowSwap
