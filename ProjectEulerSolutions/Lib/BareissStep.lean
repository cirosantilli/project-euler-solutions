import ProjectEulerSolutions.Lib.BareissState

/-! A complete Bareiss step has a nonzero divisor and exact integer quotients. -/
namespace ProjectEulerSolutions.Bareiss
set_option Elab.async false

variable {n k : Nat}

private theorem reduced_zeros (matrix : Matrix (Fin n) (Fin (n + 1)) Int) (pivot : Fin n) (previous : Int)
    (eliminated : ∀ row column, column.val < pivot.val → column.val < row.val → matrix row column = 0)
    (row : Fin n) (column : Fin (n + 1)) (processed : column.val ≤ pivot.val) (below : column.val < row.val) :
    IntegerEliminationMatrix.reduce matrix pivot pivot.castSucc previous row column = 0 := by
  by_cases kept : row.val ≤ pivot.val
  · simp only [IntegerEliminationMatrix.reduce, if_pos kept]
    exact eliminated row column (by omega) below
  · simp only [IntegerEliminationMatrix.reduce, if_neg kept, IntegerEliminationRow.reduce]
    by_cases earlier : column.val < pivot.val
    · rw [eliminated row column earlier below, eliminated pivot column earlier earlier]
      simp
    · have same : column = pivot.castSucc := Fin.ext (by simp; omega)
      subst column
      simp [mul_comm]

theorem State.exact_division {original : Matrix (Fin n) (Fin (n + 1)) Int} (state : State original k)
    (unfinished : k < n) (row : Fin n) (below : k < row.val) (column : Fin (n + 1)) :
    state.previous ∣ state.matrix ⟨k, unfinished⟩ ⟨k, by omega⟩ * state.matrix row column -
      state.matrix row ⟨k, by omega⟩ * state.matrix ⟨k, unfinished⟩ column :=
  state.active.exact_division state.previousNonzero ⟨k, unfinished⟩ ⟨k, by omega⟩ rfl rfl state.eliminated row below column

/-- Python floor division agrees with the exact integer quotient even when the previous pivot is negative. -/
theorem State.floor_update {original : Matrix (Fin n) (Fin (n + 1)) Int} (state : State original k)
    (unfinished : k < n) (row : Fin n) (below : k < row.val) (column : Fin (n + 1)) :
    (state.matrix ⟨k, unfinished⟩ ⟨k, by omega⟩ * state.matrix row column -
      state.matrix row ⟨k, by omega⟩ * state.matrix ⟨k, unfinished⟩ column).fdiv state.previous =
      IntegerEliminationMatrix.reduce state.matrix ⟨k, unfinished⟩ ⟨k, by omega⟩ state.previous row column := by
  rw [Int.fdiv_eq_ediv_of_dvd (state.exact_division unfinished row below column)]
  simp only [IntegerEliminationMatrix.reduce, show ¬row.val ≤ k by omega, if_false, IntegerEliminationRow.reduce]

/-- After placing a nonzero pivot on the diagonal, the next state preserves all invariants without assuming divisibility. -/
def State.advance {original : Matrix (Fin n) (Fin (n + 1)) Int} (state : State original k)
    (unfinished : k < n) (pivotNonzero : state.matrix ⟨k, unfinished⟩ ⟨k, by omega⟩ ≠ 0) : State original (k + 1) := by
  let pivot : Fin n := ⟨k, unfinished⟩
  let target : Fin (n + 1) := pivot.castSucc
  have divisions : ∀ row : Fin n, pivot.val < row.val → ∀ column : Fin (n + 1),
      state.previous ∣ state.matrix pivot target * state.matrix row column - state.matrix row target * state.matrix pivot column :=
    fun row below column => state.exact_division unfinished row below column
  refine {
    bounded := by omega
    matrix := IntegerEliminationMatrix.reduce state.matrix pivot target state.previous
    labels := state.labels
    prefixRows := Fin.snoc state.prefixRows (state.labels pivot)
    prefixColumns := Fin.snoc state.prefixColumns target
    active := state.active.advance state.previousNonzero pivot target rfl rfl
    previousNonzero := state.active.next_previous pivot target rfl rfl pivotNonzero
    eliminated := ?_
    nonsingular := IntegerEliminationMatrix.injective (Coefficients state.matrix) pivot state.previous state.nonsingular
      pivotNonzero state.previousNonzero (fun row below column => divisions row below column.castSucc)
    equations := fun value => (IntegerEliminationMatrix.equations_iff state.matrix pivot target state.previous
      pivotNonzero state.previousNonzero divisions value).trans (state.equations value)
    columnsOrdered := ?_
    rowsOrdered := ?_
  }
  · intro row column earlier below
    exact reduced_zeros state.matrix pivot state.previous state.eliminated row column (by dsimp [pivot]; omega) below
  · intro index
    cases index using Fin.lastCases with
    | last => simp [target, pivot]
    | cast index => simpa only [Fin.snoc_castSucc, Fin.val_castSucc] using state.columnsOrdered index
  · intro index
    cases index using Fin.lastCases with
    | last => simp [pivot]
    | cast index => simpa only [Fin.snoc_castSucc] using state.rowsOrdered index

/-- The actual pivot value becomes exactly the next previous-pivot divisor. -/
theorem State.advance_previous {original : Matrix (Fin n) (Fin (n + 1)) Int} (state : State original k)
    (unfinished : k < n) (pivotNonzero : state.matrix ⟨k, unfinished⟩ ⟨k, by omega⟩ ≠ 0) :
    (state.advance unfinished pivotNonzero).previous = state.matrix ⟨k, unfinished⟩ ⟨k, by omega⟩ :=
  (state.active ⟨k, unfinished⟩ ⟨k, by omega⟩ (Nat.le_refl k) (Nat.le_refl k)).symm

end ProjectEulerSolutions.Bareiss
