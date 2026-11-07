import ProjectEulerSolutions.Lib.BareissStep

/-! Every valid pivot choice advances elimination, and the last pivot is the permuted determinant. -/
namespace ProjectEulerSolutions.Bareiss
set_option Elab.async false

variable {n k : Nat} {original : Matrix (Fin n) (Fin (n + 1)) Int}

def Step (before : State original k) (after : State original (k + 1)) : Prop :=
  ∃ (unfinished : k < n) (candidate : Fin n) (bound : k ≤ candidate.val)
    (nonzero : (before.swap ⟨k, unfinished⟩ candidate (Nat.le_refl k) bound).matrix ⟨k, unfinished⟩ ⟨k, by omega⟩ ≠ 0),
    after = (before.swap ⟨k, unfinished⟩ candidate (Nat.le_refl k) bound).advance unfinished nonzero

theorem State.next (state : State original k) (unfinished : k < n) : ∃ after, Step state after := by
  obtain ⟨candidate, bound, nonzero⟩ := state.exists_pivot unfinished
  have swapped : (state.swap ⟨k, unfinished⟩ candidate (Nat.le_refl k) bound).matrix
      ⟨k, unfinished⟩ ⟨k, by omega⟩ ≠ 0 := by
    simpa only [State.swap, IntegerRowSwap.swap, Equiv.swap_apply_left] using nonzero
  exact ⟨(state.swap ⟨k, unfinished⟩ candidate (Nat.le_refl k) bound).advance unfinished swapped,
    unfinished, candidate, bound, swapped, rfl⟩

inductive Runs : {first last : Nat} → State original first → State original last → Prop
  | refl {phase : Nat} (state : State original phase) : Runs state state
  | next {first last : Nat} {before : State original first} {middle : State original (first + 1)} {after : State original last} :
      Step before middle → Runs middle after → Runs before after

/-- The matrix dimension bounds the number of steps; no search cutoff or convergence assumption is needed. -/
theorem State.finish (count stop : Nat) (state : State original k) (length : k + count = stop) (bound : stop ≤ n) :
    ∃ after : State original stop, Runs state after := by
  induction count generalizing k with
  | zero =>
      have same : k = stop := by omega
      subst k
      exact ⟨state, .refl state⟩
  | succ count ih =>
      obtain ⟨next, step⟩ := state.next (by omega)
      obtain ⟨after, ran⟩ := ih next (by omega)
      exact ⟨after, .next step ran⟩

theorem State.last_nonzero {size : Nat} {original : Matrix (Fin (size + 1)) (Fin (size + 2)) Int}
    (state : State original size) : state.matrix (Fin.last size) (Fin.last size).castSucc ≠ 0 := by
  obtain ⟨row, bound, nonzero⟩ := state.exists_pivot (Nat.lt_succ_self size)
  have same : row = Fin.last size := Fin.ext (by have inside := row.isLt; simp; omega)
  subst row
  exact nonzero

/-- The last diagonal is the determinant of the original coefficient matrix with the selected row permutation. -/
theorem State.last_determinant {size : Nat} {original : Matrix (Fin (size + 1)) (Fin (size + 2)) Int}
    (state : State original size) :
    state.matrix (Fin.last size) (Fin.last size).castSucc =
      ((Coefficients original).submatrix state.labels id).det := by
  rw [state.active (Fin.last size) (Fin.last size).castSucc (Nat.le_refl size) (Nat.le_refl size), BareissMinors.minor]
  have rows : Fin.snoc state.prefixRows (state.labels (Fin.last size)) = state.labels := by
    funext index
    cases index using Fin.lastCases with
    | last => simp
    | cast index => simpa only [Fin.snoc_castSucc] using state.rowsOrdered index
  have columns : Fin.snoc state.prefixColumns (Fin.last size).castSucc = Fin.castSucc := by
    funext index
    cases index using Fin.lastCases with
    | last => simp
    | cast index =>
        rw [Fin.snoc_castSucc]
        exact Fin.ext (state.columnsOrdered index)
  rw [rows, columns]
  rfl

end ProjectEulerSolutions.Bareiss
