import solvers.«83.py.lean.d».InitialRuntime
import solvers.«83.py.lean.d».Loop
import ProjectEulerSolutions.Lib.DijkstraGridTraversal

namespace solvers.«83»
open FormalSW.PyAst HeapCalls
open ProjectEulerSolutions
open RectangularMatrixCalls (MatrixRows Shape)
open TriangleBufferCalls (number encode)
open GridShortestPaths (cost finish)
set_option Elab.async false

theorem positive_result (heap : Mutable.Heap) (address : Nat) (first : Nat × List Int) (rest : MatrixRows)
    (found : heap[address]? = some (.list ((first :: rest).map fun pair => .ref pair.1)))
    (reads : ∀ pair ∈ first :: rest, heap[pair.1]? = some (.list (encode pair.2)))
    (shape : ∀ pair ∈ first :: rest, pair.2.length = first.2.length)
    (nonnegative : ∀ row ∈ (first :: rest).map Prod.snd, ∀ value ∈ row, 0 ≤ value)
    (positive : 0 < first.2.length)
    (checked : ReturnsValue program "is_rectangular" [.ref address] heap (.scalar (.bool true)) heap) :
    ∃ extra, ReturnsValue program "solve" [.ref address] heap
      (number (ProjectEulerStatements.P83.naive ((first :: rest).map Prod.snd))) (heap ++ extra) := by
  let matrix := (first :: rest).map Prod.snd
  have rowsPositive : 0 < matrix.length := by simp [matrix]
  have colsPositive : 0 < (matrix.headD []).length := positive
  have rectangular : ∀ row ∈ matrix, row.length = (matrix.headD []).length := by
    intro row member
    obtain ⟨pair, present, rfl⟩ := List.mem_map.mp member
    exact shape pair present
  let size := DijkstraGrid.total matrix
  let bound := ProjectEulerStatements.P83.bigVal matrix
  let start := ProjectEulerStatements.P83.getD2 matrix 0 0
  have sizePositive : 0 < size := Nat.mul_pos rowsPositive colsPositive
  have sizeEq : size = (rest.length + 1) * first.2.length := by simp [size, DijkstraGrid.total, matrix]
  have initialized := initial_loop_state heap address (first :: rest) found reads shape nonnegative rowsPositive colsPositive
  have locals := initial_loop_locals heap address (rest.length + 1) first.2.length bound (by rwa [← sizeEq])
  rw [← sizeEq] at locals
  have targetIndex : FiniteGridIndex.index (finish matrix rowsPositive colsPositive) =
      (first :: rest).length * first.2.length - 1 := by
    simpa only [DijkstraGrid.total, matrix, List.length_map, List.map_cons, List.headD_cons, List.length_cons] using
      DijkstraGrid.finish_index matrix rowsPositive colsPositive
  obtain ⟨shortest, value, minimal⟩ := GridShortestPaths.naive_minimum matrix rectangular nonnegative rowsPositive colsPositive
  have optimal : ∃ shortest : (DijkstraGrid.grid matrix).Walk
      (finish matrix rowsPositive colsPositive) (FiniteGridGraph.origin rowsPositive colsPositive),
      (shortest.support.map (cost matrix)).sum = ProjectEulerStatements.P83.naive matrix ∧
      ∀ walk : (DijkstraGrid.grid matrix).Walk (finish matrix rowsPositive colsPositive)
          (FiniteGridGraph.origin rowsPositive colsPositive),
        ProjectEulerStatements.P83.naive matrix ≤ (walk.support.map (cost matrix)).sum := by
    exact ⟨shortest, value, fun walk => by rw [← value]; exact minimal walk⟩
  obtain ⟨after, returned, lower, frame⟩ := loop_result address (heap.length + 1) (heap.length + 5) (heap.length + 3)
    (first :: rest) first.2.length (FiniteGridGraph.origin rowsPositive colsPositive)
    (finish matrix rowsPositive colsPositive) (ProjectEulerStatements.P83.naive matrix) nonnegative
    (FiniteGridGraph.reachable_origin rowsPositive colsPositive) targetIndex optimal
    (initialHeap heap size bound start)
    (initialEnv (totalEnv address (rest.length + 1) first.2.length) heap size bound)
    (initialDistances size bound start) (List.replicate size false) [.ref (heap.length + 2)]
    initialized locals (by simp [List.getD])
  have retained : heap <+: after := List.prefix_iff_getElem?.mpr (by
    intro i inside
    have initialInside : i < (initialHeap heap size bound start).length := by
      rw [initial_heap_layout]
      simp only [List.length_append, List.length_cons, List.length_nil]
      omega
    rw [frame i initialInside (by omega) (by omega) (by omega), initial_heap_layout,
      List.getElem?_append_left inside, List.getElem?_eq_getElem inside])
  obtain ⟨extra, afterEq⟩ := List.prefix_iff_exists_eq_append.mp retained
  have startRead : start = first.2[0] := by
    simpa [start, matrix] using matrix_cost_read (first :: rest) 0 0 (by simp) positive
  rw [startRead, afterEq] at returned
  refine ⟨extra, positive_result_of_loop heap (heap ++ extra) address first rest _ found reads positive checked ?_⟩
  simpa only [sizeEq] using returned

theorem solve.result (heap : Mutable.Heap) (address : Nat) (rows : MatrixRows)
    (found : heap[address]? = some (.list (rows.map fun pair => .ref pair.1)))
    (reads : ∀ pair ∈ rows, heap[pair.1]? = some (.list (encode pair.2)))
    (shape : Shape rows)
    (nonnegative : ∀ row ∈ rows.map Prod.snd, ∀ value ∈ row, 0 ≤ value)
    (checked : ReturnsValue program "is_rectangular" [.ref address] heap (.scalar (.bool true)) heap) :
    ∃ extra, ReturnsValue program "solve" [.ref address] heap
      (number (ProjectEulerStatements.P83.naive (rows.map Prod.snd))) (heap ++ extra) := by
  cases rows with
  | nil =>
      refine ⟨[], ?_⟩
      simpa only [List.map_nil, GridRelaxationAgreement.naive_empty, List.append_nil] using empty_result heap address found
  | cons first rest =>
      by_cases zero : first.2.length = 0
      · refine ⟨[], ?_⟩
        rw [GridRelaxationAgreement.naive_zero_width _ (by simpa using zero), List.append_nil]
        exact zero_width_result heap address first rest found (reads first (by simp)) zero checked
      · have rectangular : ∀ pair ∈ first :: rest, pair.2.length = first.2.length := by
          intro pair member
          obtain ⟨i, inside, rfl⟩ := List.mem_iff_getElem.mp member
          exact shape i inside
        exact positive_result heap address first rest found reads rectangular nonnegative (by omega) checked

end solvers.«83»
