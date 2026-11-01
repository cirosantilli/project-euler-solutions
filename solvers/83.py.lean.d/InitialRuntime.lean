import solvers.«83.py.lean.d».RuntimeState
import solvers.«83.py.lean.d».CoverageInitial

namespace solvers.«83»
open FormalSW.PyAst HeapCalls
open ProjectEulerSolutions
open RectangularMatrixCalls (MatrixRows)
open TriangleBufferCalls (number encode)
open GridShortestPaths (cost)
set_option Elab.async false

theorem initial_loop_state (heap : Mutable.Heap) (address : Nat) (rows : MatrixRows)
    (found : heap[address]? = some (.list (rows.map fun pair => .ref pair.1)))
    (reads : ∀ pair ∈ rows, heap[pair.1]? = some (.list (encode pair.2)))
    (shape : ∀ pair ∈ rows, pair.2.length = ((rows.map Prod.snd).headD []).length)
    (nonnegative : ∀ row ∈ rows.map Prod.snd, ∀ value ∈ row, 0 ≤ value)
    (rowsPositive : 0 < (rows.map Prod.snd).length)
    (colsPositive : 0 < ((rows.map Prod.snd).headD []).length) :
    let matrix := rows.map Prod.snd
    let size := DijkstraGrid.total matrix
    let bound := ProjectEulerStatements.P83.bigVal matrix
    let start := ProjectEulerStatements.P83.getD2 matrix 0 0
    LoopState (initialHeap heap size bound start) address (heap.length + 1) (heap.length + 5) (heap.length + 3)
      rows (matrix.headD []).length (FiniteGridGraph.origin rowsPositive colsPositive)
      (initialDistances size bound start) (List.replicate size false) [.ref (heap.length + 2)] := by
  intro matrix size bound start
  have rectangular : ∀ row ∈ matrix, row.length = (matrix.headD []).length := by
    intro row member
    obtain ⟨pair, present, rfl⟩ := List.mem_map.mp member
    exact shape pair present
  have positive : 0 < size := Nat.mul_pos rowsPositive colsPositive
  have finite : start < bound := by
    exact DijkstraGrid.cost_lt_bound matrix rectangular nonnegative (FiniteGridGraph.origin rowsPositive colsPositive)
  have addressInside := (List.getElem?_eq_some_iff.mp found).1
  have rowsInside (pair : Nat × List Int) (member : pair ∈ rows) : pair.1 < heap.length :=
    (List.getElem?_eq_some_iff.mp (reads pair member)).1
  have queueTyped := initial_queue_typed heap size bound start
  refine ⟨?_, ?_, by omega, by omega, ?_, ?_, rfl, ?_⟩
  · refine ⟨initial_heap_preserves heap size bound start address _ found,
      fun pair member => initial_heap_preserves heap size bound start pair.1 _ (reads pair member),
      by omega, ?_, by omega, shape, ?_, ?_, initial_distances_found heap size bound start,
      initial_queue_found heap size bound start, ?_, queueTyped.2.2, ?_⟩
    · intro pair member
      have inside := rowsInside pair member
      omega
    · simp only [initialDistances, List.length_set, List.length_replicate, size, DijkstraGrid.total, matrix, List.length_map]
    · intro i inside
      have read := initial_distance_read size i bound start positive
      rw [List.getD_eq_getElem _ _ inside] at read
      rw [read]
      change (if i = 0 then start else bound) ≤ bound
      split <;> omega
    · intro value member
      obtain rfl := List.mem_singleton.mp member
      exact queueTyped.1
    · simpa only [initialDistances, List.length_set, List.length_replicate, distanceValues] using
        initial_covered heap size bound start positive finite
  · simpa only [encodeVisited, List.map_replicate] using initial_visited_found heap size bound start
  · intro pair member
    have inside := rowsInside pair member
    omega
  · simp only [initialDistances, List.length_set, List.length_replicate]
  · have invariant := DijkstraGrid.initial_invariant matrix rectangular nonnegative rowsPositive colsPositive
    rw [← initial_buffers_model matrix] at invariant
    exact invariant

theorem initial_loop_locals (heap : Mutable.Heap) (address height width : Nat) (bound : Int)
    (positive : 0 < height * width) :
    LoopLocals (initialEnv (totalEnv address height width) heap (height * width) bound)
      address (heap.length + 1) (heap.length + 5) (heap.length + 3) height width := by
  refine ⟨rfl, rfl, rfl, rfl, rfl, rfl, ?_⟩
  simp [initialEnv, setLocal, Int.ofNat_sub (by omega : 1 ≤ height * width)]

end solvers.«83»
