import solvers.«84.py.lean.d».StationaryRow

namespace solvers.«84».stationary_weights
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
open TriangleBufferCalls (encode)
set_option Elab.async false

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

end solvers.«84».stationary_weights
