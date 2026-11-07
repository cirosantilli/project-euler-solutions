import solvers.«84.py.lean.d».StationaryNormalization

namespace solvers.«84».stationary_weights
open FormalSW.PyAst ProjectEulerSolutions
open TriangleBufferCalls (encode)
set_option Elab.async false

def augmentedRow (sides : Nat) (row : Fin 120) : List Int :=
  columnValues sides row 0 120 ++ [Monopoly.rightHandSide row]

theorem augmentedRow_length (sides : Nat) (row : Fin 120) : (augmentedRow sides row).length = 121 := by
  simp [augmentedRow, columnValues]

theorem augmentedRow_coefficient (sides : Nat) (row column : Fin 120) :
    (augmentedRow sides row)[column.val]? = some (Monopoly.coefficients sides row column) := by
  rw [augmentedRow, List.getElem?_append_left (by simp [columnValues])]
  simp [columnValues, column.isLt]

theorem augmentedRow_constant (sides : Nat) (row : Fin 120) :
    (augmentedRow sides row)[120]? = some (Monopoly.rightHandSide row) := by
  simp [augmentedRow, columnValues]

theorem augmentedRow_ordinary (sides : Nat) (row : Fin 120) (ordinary : row ≠ 119) :
    augmentedRow sides row = rowValues sides row := by
  simp [augmentedRow, rowValues, Monopoly.rightHandSide, ordinary]

theorem augmentedRow_normalization (sides : Nat) : augmentedRow sides 119 = List.replicate 121 1 := by
  simp [augmentedRow, columnValues, Monopoly.coefficients, Monopoly.rightHandSide, List.replicate_succ']

def MatrixView (sides : Nat) (heap : Mutable.Heap) (root : Nat) : Prop :=
  ∃ items, heap[root]? = some (.list items) ∧ items.length = 120 ∧
    ∀ row : Fin 120, ∃ address, items[row.val]? = some (.ref address) ∧
      heap[address]? = some (.list (encode (augmentedRow sides row)))

/-- Every row and column of the source-created matrix agrees with the nonsingular integer system. -/
theorem prepared_view (base : Mutable.Heap) (sides : Nat) : MatrixView sides (preparedHeap base sides) base.length := by
  refine ⟨rowReferences base 119 ++ [.ref ((matrixHeap base sides 119).length + 3)],
    prepared_found base sides, by simp [rowReferences], ?_⟩
  intro row
  by_cases last : row = 119
  · subst row
    refine ⟨(matrixHeap base sides 119).length + 3, ?_, ?_⟩
    · simp [rowReferences]
    · rw [augmentedRow_normalization]
      exact prepared_normalization_found base sides
  · have below : row.val < 119 := by
      have different : row.val ≠ 119 := by intro same; apply last; exact Fin.ext same
      have bound := row.isLt
      omega
    refine ⟨base.length + 3 * row.val + 3, ?_, ?_⟩
    · rw [List.getElem?_append_left (by simpa [rowReferences] using below)]
      simp [rowReferences, below]
    · rw [augmentedRow_ordinary sides row last]
      simpa using prepared_row_found base sides row.val below

end solvers.«84».stationary_weights
