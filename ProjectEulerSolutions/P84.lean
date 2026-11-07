import ProjectEulerStatements.P84
namespace ProjectEulerSolutions.P84
open ProjectEulerSolutions Bareiss

/-- Select the first nonzero active pivot; the nonsingular invariant guarantees one exists. -/
def pivot {n k : Nat} {original : Matrix (Fin n) (Fin (n+1)) Int}
    (state : State original k) (unfinished : k < n) :
    { row : Fin n // k ≤ row.val ∧ state.matrix row ⟨k, by omega⟩ ≠ 0 } := by
  let candidates := Finset.univ.filter (fun row : Fin n => k ≤ row.val ∧ state.matrix row ⟨k, by omega⟩ ≠ 0)
  have nonempty : candidates.Nonempty := by
    obtain ⟨row, lower, nonzero⟩ := state.exists_pivot unfinished
    exact ⟨row, by simp [candidates, lower, nonzero]⟩
  exact ⟨candidates.min' nonempty, (Finset.mem_filter.mp (Finset.min'_mem candidates nonempty)).2⟩

/-- Store every entry once between elimination stages. -/
def cache {n m : Nat} (matrix : Matrix (Fin n) (Fin m) Int) : Matrix (Fin n) (Fin m) Int :=
  let entries := Array.ofFn (fun row => Array.ofFn (matrix row))
  fun row column => (entries[row.val]!)[column.val]!

@[simp] theorem cache_eq {n m : Nat} (matrix : Matrix (Fin n) (Fin m) Int) : cache matrix = matrix := by
  funext row column
  simp [cache]

def withMatrix {n k : Nat} {original : Matrix (Fin n) (Fin (n+1)) Int}
    (state : State original k) (matrix : Matrix (Fin n) (Fin (n+1)) Int)
    (same : matrix = state.matrix) : State original k :=
  { state with matrix := matrix
               active := by rw [same]; exact state.active
               eliminated := by rw [same]; exact state.eliminated
               nonsingular := by rw [same]; exact state.nonsingular
               equations := by rw [same]; exact state.equations }

theorem withMatrix_previous {n k : Nat} {original : Matrix (Fin n) (Fin (n+1)) Int}
    (state : State original k) (matrix : Matrix (Fin n) (Fin (n+1)) Int) (same : matrix = state.matrix) :
    (withMatrix state matrix same).previous = state.previous := rfl

def advance {n k : Nat} {original : Matrix (Fin n) (Fin (n+1)) Int}
    (state : State original k) (unfinished : k < n) (previous : Int) (agrees : state.previous = previous) :
    { next : State original (k+1) × Int // next.1.previous = next.2 } := by
  let selected := pivot state unfinished
  let swapped := state.swap ⟨k, unfinished⟩ selected.val (Nat.le_refl k) selected.property.1
  have nonzero : swapped.matrix ⟨k, unfinished⟩ ⟨k, by omega⟩ ≠ 0 := by
    simpa [swapped, State.swap, IntegerRowSwap.swap] using selected.property.2
  let next := swapped.advance unfinished nonzero
  let matrix := cache (IntegerEliminationMatrix.reduce swapped.matrix ⟨k, unfinished⟩ ⟨k, by omega⟩ previous)
  have same : matrix = next.matrix := by
    simp only [matrix, cache_eq, next, State.advance]
    congr 1
    exact agrees.symm
  let certified := withMatrix next matrix same
  let stored : State original (k+1) := {
    bounded := certified.bounded
    matrix := matrix
    labels := swapped.labels
    prefixRows := Fin.snoc swapped.prefixRows (swapped.labels ⟨k, unfinished⟩)
    prefixColumns := Fin.snoc swapped.prefixColumns (⟨k, by omega⟩ : Fin (n+1))
    active := certified.active
    previousNonzero := certified.previousNonzero
    eliminated := certified.eliminated
    nonsingular := certified.nonsingular
    equations := certified.equations
    columnsOrdered := certified.columnsOrdered
    rowsOrdered := certified.rowsOrdered }

  exact ⟨(stored, swapped.matrix ⟨k, unfinished⟩ ⟨k, by omega⟩),
    swapped.advance_previous unfinished nonzero⟩

def eliminate {n k : Nat} {original : Matrix (Fin n) (Fin (n+1)) Int}
    (state : State original k) (previous : Int) (agrees : state.previous = previous) :
    (remaining : Nat) → k + remaining ≤ n → State original (k+remaining)
  | 0, _ => state
  | remaining+1, bound => by
    let next := advance state (by omega) previous agrees
    have finished := eliminate next.val.1 next.val.2 next.property remaining (by omega)
    simpa only [Nat.add_assoc, Nat.add_comm 1 remaining] using finished

def finalState (sides : Nat) (positive : 0 < sides) : State (Monopoly.augmentedMatrix sides) 119 :=
  let initial := Monopoly.initialElimination sides positive
  let stored := withMatrix initial (cache initial.matrix) (cache_eq _)
  let state : State (Monopoly.augmentedMatrix sides) 119 :=
    eliminate (n := 120) (k := 0) stored initial.previous (withMatrix_previous _ _ _) 119 (by decide)
  state

def weights (sides : Nat) (positive : 0 < sides) : Fin 120 → Int :=
  let state := finalState sides positive
  fun index => if state.matrix 119 119 < 0 then -state.solution index else state.solution index

/-- Materialize the solved vector once before ranking the forty board squares. -/
def weightArray (sides : Nat) (positive : 0 < sides) : Array Int :=
  let state := finalState sides positive
  let solution := state.solution
  let negative := state.matrix 119 119 < 0
  Array.ofFn (fun index : Fin 120 => if negative then -solution index else solution index)

theorem weightArray_get (sides : Nat) (positive : 0 < sides) (index : Fin 120) :
    (weightArray sides positive)[index.val]! = weights sides positive index := by
  change (Array.ofFn (weights sides positive))[index.val]! = _
  simp

def solve (sides : Nat) (positive : 0 < sides) : String :=
  let values := weightArray sides positive
  Monopoly.modal (fun index => values[index.val]!)
end ProjectEulerSolutions.P84
