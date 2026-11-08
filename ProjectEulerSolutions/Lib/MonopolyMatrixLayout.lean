import ProjectEulerSolutions.Lib.MonopolyBareiss
import ProjectEulerSolutions.Lib.TriangleBufferCalls
import ProjectEulerSolutions.Lib.HeapCollection
import ProjectEulerSolutions.Lib.MutableIntegerMatrix

/-! The shared augmented-matrix heap layout, independent of either Python module. -/
namespace ProjectEulerSolutions.MonopolyMatrixLayout
open FormalSW.PyAst HeapCalls
open TriangleBufferCalls (encode)
set_option Elab.async false

def columnValues (sides : Nat) (target : Fin 120) (first count : Nat) : List Int :=
  (List.range' first count).map fun source => Monopoly.coefficients sides target (Fin.ofNat 120 source)

def rowValues (sides : Nat) (target : Fin 120) : List Int := columnValues sides target 0 120 ++ [0]

def rowObjects (sides : Nat) (target : Fin 120) : Mutable.Heap :=
  [.list (encode (columnValues sides target 0 120)), .list [.scalar (.int 0)], .list (encode (rowValues sides target))]

def rowHeap (heap : Mutable.Heap) (sides : Nat) (target : Fin 120) : Mutable.Heap := heap ++ rowObjects sides target

theorem row_found (heap : Mutable.Heap) (sides : Nat) (target : Fin 120) :
    (rowHeap heap sides target)[heap.length + 2]? = some (.list (encode (rowValues sides target))) := by
  simp [rowHeap, rowObjects]

theorem row_length (sides : Nat) (target : Fin 120) : (rowValues sides target).length = 121 := by
  simp [rowValues, columnValues]

theorem row_coefficient (sides : Nat) (target column : Fin 120) :
    (rowValues sides target)[column.val]? = some (Monopoly.coefficients sides target column) := by
  rw [rowValues, List.getElem?_append_left (by simp [columnValues])]
  simp [columnValues, column.isLt]

theorem row_constant (sides : Nat) (target : Fin 120) : (rowValues sides target)[120]? = some 0 := by
  simp [rowValues, columnValues]

def rowReferences (base : Mutable.Heap) (count : Nat) : List Mutable.Value :=
  (List.range count).map fun row => .ref (base.length + 3 * row + 3)

def matrixHeap (base : Mutable.Heap) (sides count : Nat) : Mutable.Heap :=
  HeapCollection.working base (rowReferences base count)
    ((List.range count).flatMap fun row => rowObjects sides (Fin.ofNat 120 row))

theorem matrix_length (base : Mutable.Heap) (sides count : Nat) :
    (matrixHeap base sides count).length = base.length + 1 + 3 * count := by
  simp [matrixHeap, HeapCollection.working, List.length_flatMap, rowObjects, Nat.mul_comm]
  omega

theorem references_next (base : Mutable.Heap) (sides count : Nat) :
    rowReferences base (count + 1) = rowReferences base count ++ [.ref ((matrixHeap base sides count).length + 2)] := by
  simp [rowReferences, List.range_succ, matrix_length]
  omega

theorem matrix_next (base : Mutable.Heap) (sides count : Nat) :
    (rowHeap (matrixHeap base sides count) sides (Fin.ofNat 120 count)).set base.length
      (.list (rowReferences base count ++ [.ref ((matrixHeap base sides count).length + 2)])) =
      matrixHeap base sides (count + 1) := by
  rw [← references_next]
  simp [rowHeap, matrixHeap, HeapCollection.append, HeapCollection.store, List.range_succ]

theorem matrix_found (base : Mutable.Heap) (sides count : Nat) :
    (matrixHeap base sides count)[base.length]? = some (.list (rowReferences base count)) :=
  HeapCollection.found _ _ _

theorem matrix_row_found (base : Mutable.Heap) (sides count row : Nat) (inside : row < count) :
    (matrixHeap base sides count)[base.length + 3 * row + 3]? =
      some (.list (encode (rowValues sides (Fin.ofNat 120 row)))) := by
  induction count with
  | zero => omega
  | succ count ih =>
      rw [← matrix_next]
      rw [List.getElem?_set_ne (by omega)]
      by_cases last : row = count
      · subst row
        have atRow := row_found (matrixHeap base sides count) sides (Fin.ofNat 120 count)
        simpa only [matrix_length, show base.length + 1 + 3 * count + 2 = base.length + 3 * count + 3 by omega] using atRow
      · exact HeapCollection.prefix_found _ _ _ _ (ih (by omega))

def normalizationExpression : Expr :=
  .binOp (.binOp (.list [.constant (.int 1)]) .mult (.name "size")) .add (.list [.constant (.int 1)])

def normalizationObjects (size : Nat) : Mutable.Heap :=
  [.list [.scalar (.int 1)], .list (encode (List.replicate size 1)),
    .list [.scalar (.int 1)], .list (encode (List.replicate (size + 1) 1))]

def preparedHeap (base : Mutable.Heap) (sides : Nat) : Mutable.Heap :=
  (matrixHeap base sides 119 ++ normalizationObjects 120).set base.length
    (.list (rowReferences base 119 ++ [.ref ((matrixHeap base sides 119).length + 3)]))

theorem prepared_prefix (base : Mutable.Heap) (sides : Nat) : ∃ extra, preparedHeap base sides = base ++ extra := by
  unfold preparedHeap matrixHeap
  rw [HeapCollection.append, HeapCollection.store]
  exact ⟨_, rfl⟩

theorem prepared_found (base : Mutable.Heap) (sides : Nat) :
    (preparedHeap base sides)[base.length]? =
      some (.list (rowReferences base 119 ++ [.ref ((matrixHeap base sides 119).length + 3)])) := by
  apply List.getElem?_set_self
  simp [matrix_length, normalizationObjects]
  omega

theorem prepared_row_found (base : Mutable.Heap) (sides row : Nat) (inside : row < 119) :
    (preparedHeap base sides)[base.length + 3 * row + 3]? =
      some (.list (encode (rowValues sides (Fin.ofNat 120 row)))) := by
  unfold preparedHeap
  rw [List.getElem?_set_ne (by omega)]
  exact HeapCollection.prefix_found _ _ _ _ (matrix_row_found base sides 119 row inside)

theorem prepared_normalization_found (base : Mutable.Heap) (sides : Nat) :
    (preparedHeap base sides)[(matrixHeap base sides 119).length + 3]? =
      some (.list (encode (List.replicate 121 1))) := by
  unfold preparedHeap
  rw [List.getElem?_set_ne (by rw [matrix_length]; omega)]
  simp [normalizationObjects]

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

def setupEnv (env : HeapCalls.Env) (heap : Mutable.Heap) : HeapCalls.Env :=
  setLocal (setLocal env "size" (.scalar (.int 120))) "matrix" (.ref heap.length)

theorem initial_entry (sides : Nat) (positive : 0 < sides) (row : Fin 120) (column : Fin 121) :
    (augmentedRow sides row)[column.val]? = some ((Monopoly.initialElimination sides positive).matrix row column) := by
  change _ = some (Monopoly.augmentedMatrix sides row column)
  cases column using Fin.lastCases with
  | last =>
      simpa only [Monopoly.augmentedMatrix, Fin.snoc_last] using augmentedRow_constant sides row
  | cast column =>
      simpa only [Monopoly.augmentedMatrix, Fin.snoc_castSucc] using augmentedRow_coefficient sides row column

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

def rowAddresses (base : Mutable.Heap) (row : Fin 120) : Nat :=
  base.length + 3 * row.val + 3 + if row = 119 then 1 else 0

theorem rowAddresses_distinct (base : Mutable.Heap) : Function.Injective (rowAddresses base) := by
  intro first second same
  apply Fin.ext
  unfold rowAddresses at same
  split_ifs at same <;> omega

theorem rowAddresses_after (base : Mutable.Heap) (row : Fin 120) : base.length < rowAddresses base row := by
  unfold rowAddresses
  split_ifs <;> omega

theorem prepared_references (base : Mutable.Heap) (sides : Nat) :
    rowReferences base 119 ++ [.ref ((matrixHeap base sides 119).length + 3)] =
      List.ofFn (fun row => Mutable.Value.ref (rowAddresses base row)) := by
  apply List.ext_getElem
  · simp only [List.length_append, rowReferences, List.length_map, List.length_range, List.length_singleton, List.length_ofFn]
  · intro index left right
    have inside : index < 120 := by simpa only [List.length_ofFn] using right
    rw [List.getElem_ofFn]
    by_cases last : index = 119
    · subst index
      simp [rowReferences, rowAddresses, matrix_length]
    · have below : index < 119 := by omega
      have different : (⟨index, inside⟩ : Fin 120) ≠ 119 := by intro equal; exact last (congrArg Fin.val equal)
      rw [List.getElem_append_left (show index < (rowReferences base 119).length by simpa [rowReferences] using below)]
      simp [rowReferences, rowAddresses, different]

theorem augmentedRow_ofFn (sides : Nat) (positive : 0 < sides) (row : Fin 120) :
    TriangleBufferCalls.encode (augmentedRow sides row) =
      List.ofFn (fun column => Mutable.Value.scalar (.int ((Monopoly.initialElimination sides positive).matrix row column))) := by
  apply List.ext_getElem
  · simp only [TriangleBufferCalls.encode, List.length_map, augmentedRow_length, List.length_ofFn]
  · intro index left right
    have inside : index < 121 := by simpa only [List.length_ofFn] using right
    have read := initial_entry sides positive row ⟨index, inside⟩
    have entry := (List.getElem?_eq_some_iff.mp read).2
    simpa only [TriangleBufferCalls.encode, List.getElem_map, List.getElem_ofFn] using
      congrArg (fun value => Mutable.Value.scalar (.int value)) entry

theorem prepared_mutable_view (base : Mutable.Heap) (sides : Nat) (positive : 0 < sides) :
    MutableIntegerMatrix.View (Monopoly.initialElimination sides positive).matrix
      (preparedHeap base sides) base.length (rowAddresses base) := by
  refine ⟨?_, ?_, rowAddresses_distinct base, fun row => Nat.ne_of_gt (rowAddresses_after base row)⟩
  · rw [← prepared_references base sides]
    exact prepared_found base sides
  · intro row
    rw [← augmentedRow_ofFn sides positive row]
    by_cases last : row = 119
    · subst row
      rw [augmentedRow_normalization]
      simpa [rowAddresses, matrix_length] using prepared_normalization_found base sides
    · have below : row.val < 119 := by
        have different : row.val ≠ 119 := by intro equal; exact last (Fin.ext equal)
        have inside := row.isLt
        omega
      rw [augmentedRow_ordinary sides row last]
      simpa [rowAddresses, last] using prepared_row_found base sides row.val below

end ProjectEulerSolutions.MonopolyMatrixLayout
