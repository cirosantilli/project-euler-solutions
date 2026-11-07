import ProjectEulerSolutions.Lib.BareissInvariant
import ProjectEulerSolutions.Lib.IntegerRowSwap
import ProjectEulerSolutions.Lib.GaussianPivot

/-! Invariants for the complete augmented integer elimination, including pivot row permutations. -/
namespace ProjectEulerSolutions.Bareiss
open scoped BigOperators
set_option Elab.async false

variable {n k : Nat}

abbrev Coefficients (matrix : Matrix (Fin n) (Fin (n + 1)) Int) : Matrix (Fin n) (Fin n) Int :=
  fun row column => matrix row column.castSucc

def Equations (matrix : Matrix (Fin n) (Fin (n + 1)) Int) (value : Fin (n + 1) → ℚ) : Prop :=
  ∀ row, ∑ column, (matrix row column : ℚ) * value column = 0

structure State (original : Matrix (Fin n) (Fin (n + 1)) Int) (processed : Nat) where
  bounded : processed ≤ n
  matrix : Matrix (Fin n) (Fin (n + 1)) Int
  labels : Equiv.Perm (Fin n)
  prefixRows : Fin processed → Fin n
  prefixColumns : Fin processed → Fin (n + 1)
  active : BareissInvariant.Active original matrix prefixRows prefixColumns labels
  previousNonzero : BareissMinors.minor original prefixRows prefixColumns ≠ 0
  eliminated : ∀ row column, column.val < processed → column.val < row.val → matrix row column = 0
  nonsingular : Function.Injective (Matrix.mulVec (fun row column => (Coefficients matrix row column : ℚ)))
  equations : ∀ value, Equations matrix value ↔ Equations original value
  columnsOrdered : ∀ index, (prefixColumns index).val = index.val
  rowsOrdered : ∀ index, prefixRows index = labels ⟨index.val, Nat.lt_of_lt_of_le index.isLt bounded⟩

def State.previous {original : Matrix (Fin n) (Fin (n + 1)) Int} (state : State original k) : Int :=
  BareissMinors.minor original state.prefixRows state.prefixColumns

def initial (original : Matrix (Fin n) (Fin (n + 1)) Int)
    (nonsingular : Function.Injective (Matrix.mulVec (fun row column => (Coefficients original row column : ℚ)))) :
    State original 0 where
  bounded := Nat.zero_le n
  matrix := original
  labels := Equiv.refl _
  prefixRows := Fin.elim0
  prefixColumns := Fin.elim0
  active := BareissInvariant.initial original
  previousNonzero := by rw [BareissInvariant.initial_previous]; decide
  eliminated := by intro row column earlier _; omega
  nonsingular := nonsingular
  equations := fun _ => Iff.rfl
  columnsOrdered := by intro index; exact Fin.elim0 index
  rowsOrdered := by intro index; exact Fin.elim0 index

/-- Swapping active rows keeps the selected prefix and merely permutes the remaining original-row labels. -/
def State.swap {original : Matrix (Fin n) (Fin (n + 1)) Int} (state : State original k)
    (first second : Fin n) (firstBound : k ≤ first.val) (secondBound : k ≤ second.val) : State original k where
  bounded := state.bounded
  matrix := IntegerRowSwap.swap state.matrix first second
  labels := (Equiv.swap first second).trans state.labels
  prefixRows := state.prefixRows
  prefixColumns := state.prefixColumns
  active := state.active.reindex (Equiv.swap first second)
    (fun row rowBound => IntegerRowSwap.stays_above first second row firstBound secondBound rowBound)
  previousNonzero := state.previousNonzero
  eliminated := IntegerRowSwap.zeros state.matrix first second firstBound secondBound state.eliminated
  nonsingular := IntegerRowSwap.injective (Coefficients state.matrix) first second state.nonsingular
  equations := fun value => (IntegerRowSwap.equations_iff state.matrix first second value).trans (state.equations value)
  columnsOrdered := state.columnsOrdered
  rowsOrdered := by
    intro index
    rw [state.rowsOrdered]
    let originalIndex : Fin n := ⟨index.val, Nat.lt_of_lt_of_le index.isLt state.bounded⟩
    have firstNe : originalIndex ≠ first := by
      intro equal
      have same := congrArg Fin.val equal
      change index.val = first.val at same
      omega
    have secondNe : originalIndex ≠ second := by
      intro equal
      have same := congrArg Fin.val equal
      change index.val = second.val at same
      omega
    change state.labels originalIndex = state.labels (Equiv.swap first second originalIndex)
    rw [Equiv.swap_apply_of_ne_of_ne firstNe secondNe]

/-- Every unfinished nonsingular state admits a pivot in the active part of its next column. -/
theorem State.exists_pivot {original : Matrix (Fin n) (Fin (n + 1)) Int} (state : State original k) (unfinished : k < n) :
    ∃ row : Fin n, k ≤ row.val ∧ state.matrix row (⟨k, by omega⟩ : Fin (n + 1)) ≠ 0 := by
  obtain ⟨row, bound, nonzero⟩ := GaussianPivot.exists_pivot_of_injective
    (fun row column => (Coefficients state.matrix row column : ℚ)) state.nonsingular ⟨k, unfinished⟩
    (by
      intro row previous earlier below
      change (state.matrix row previous.castSucc : ℚ) = 0
      exact_mod_cast state.eliminated row previous.castSucc earlier below)
  exact ⟨row, bound, by exact_mod_cast nonzero⟩

end ProjectEulerSolutions.Bareiss
