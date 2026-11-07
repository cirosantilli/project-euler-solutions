import solvers.«84.py.lean.d».StationaryEliminationRowResult

namespace solvers.«84».stationary_weights
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
set_option Elab.async false

def columnRows (matrix : Matrix (Fin 120) (Fin 121) Int) (pivot : Fin 120) (previous : Int) (stop : Nat) :
    Matrix (Fin 120) (Fin 121) Int :=
  fun row column => if pivot.val < row.val ∧ row.val < stop then reducedRow matrix pivot row previous column else matrix row column

theorem columnRows_start (matrix : Matrix (Fin 120) (Fin 121) Int) (pivot : Fin 120) (previous : Int) :
    columnRows matrix pivot previous (pivot.val + 1) = matrix := by
  funext row column
  simp only [columnRows, show ¬(pivot.val < row.val ∧ row.val < pivot.val + 1) by omega, if_false]

theorem columnRows_pivot (matrix : Matrix (Fin 120) (Fin 121) Int) (pivot : Fin 120) (previous : Int) (stop : Nat) :
    columnRows matrix pivot previous stop pivot = matrix pivot := by
  funext column
  simp [columnRows]

theorem columnRows_current (matrix : Matrix (Fin 120) (Fin 121) Int) (pivot row : Fin 120) (previous : Int) :
    columnRows matrix pivot previous row.val row = matrix row := by
  funext column
  simp [columnRows]

theorem columnRows_reduction (matrix : Matrix (Fin 120) (Fin 121) Int) (pivot row : Fin 120) (previous : Int) :
    reducedRow (columnRows matrix pivot previous row.val) pivot row previous = reducedRow matrix pivot row previous := by
  funext column
  simp only [reducedRow, columnRows_pivot, columnRows_current]

theorem columnRows_step (matrix : Matrix (Fin 120) (Fin 121) Int) (pivot row : Fin 120)
    (previous : Int) (below : pivot.val < row.val) :
    Function.update (columnRows matrix pivot previous row.val) row (reducedRow matrix pivot row previous) =
      columnRows matrix pivot previous (row.val + 1) := by
  funext index column
  by_cases same : index = row
  · subst index
    simp [columnRows, below]
  · rw [Function.update_of_ne same]
    have different : index.val ≠ row.val := fun equal => same (Fin.ext equal)
    have comparison : index.val < row.val + 1 ↔ index.val < row.val := by omega
    simp only [columnRows, comparison]

theorem columnRows_finished {sides k : Nat} (state : Bareiss.State (Monopoly.augmentedMatrix sides) k)
    (unfinished : k < 120) (nonzero : state.matrix ⟨k, unfinished⟩ ⟨k, by omega⟩ ≠ 0) :
    columnRows state.matrix ⟨k, unfinished⟩ state.previous 120 = (state.advance unfinished nonzero).matrix := by
  funext row column
  by_cases below : k < row.val
  · simp only [columnRows, below, row.isLt, and_self, if_true, reducedRow]
    exact state.floor_update unfinished row below column
  · simp only [columnRows, below, false_and, if_false, Bareiss.State.advance, IntegerEliminationMatrix.reduce,
      show row.val ≤ k by omega, if_true]

end solvers.«84».stationary_weights
