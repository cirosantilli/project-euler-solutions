import solvers.«84.py.lean.d».StationarySetup
import ProjectEulerSolutions.Lib.MonopolyBareiss

namespace solvers.«84».stationary_weights
open FormalSW.PyAst ProjectEulerSolutions
set_option Elab.async false

theorem initial_entry (sides : Nat) (positive : 0 < sides) (row : Fin 120) (column : Fin 121) :
    (augmentedRow sides row)[column.val]? = some ((Monopoly.initialElimination sides positive).matrix row column) := by
  change _ = some (Monopoly.augmentedMatrix sides row column)
  cases column using Fin.lastCases with
  | last =>
      simpa only [Monopoly.augmentedMatrix, Fin.snoc_last] using augmentedRow_constant sides row
  | cast column =>
      simpa only [Monopoly.augmentedMatrix, Fin.snoc_castSucc] using augmentedRow_coefficient sides row column

/-- The source-created heap supplies every entry of the initial state whose Bareiss invariants are proved. -/
theorem MatrixView.initial_entries {sides : Nat} {heap : Mutable.Heap} {root : Nat}
    (view : MatrixView sides heap root) (positive : 0 < sides) :
    ∃ items, heap[root]? = some (.list items) ∧ items.length = 120 ∧
      ∀ row : Fin 120, ∃ (address : Nat) (values : List Int), items[row.val]? = some (.ref address) ∧
        heap[address]? = some (.list (TriangleBufferCalls.encode values)) ∧ values.length = 121 ∧
        ∀ column : Fin 121, values[column.val]? = some ((Monopoly.initialElimination sides positive).matrix row column) := by
  obtain ⟨items, found, length, rows⟩ := view
  refine ⟨items, found, length, ?_⟩
  intro row
  obtain ⟨address, selected, contents⟩ := rows row
  exact ⟨address, augmentedRow sides row, selected, contents, augmentedRow_length sides row, initial_entry sides positive row⟩

end solvers.«84».stationary_weights
