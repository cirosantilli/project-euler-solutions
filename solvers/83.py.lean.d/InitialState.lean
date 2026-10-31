import solvers.«83.py.lean.d».Setup
import ProjectEulerSolutions.Lib.DijkstraGridInitial

namespace solvers.«83»
open FormalSW.PyAst
open ProjectEulerSolutions.TriangleBufferCalls (number encode)
open ProjectEulerSolutions.BinaryHeapOrder (Ordered)
set_option Elab.async false

theorem initial_heap_layout (heap : Mutable.Heap) (size : Nat) (bound distance : Int) :
    initialHeap heap size bound distance = heap ++
      [.list [number bound], .list (encode (initialDistances size bound distance)),
       .tuple [number distance, number 0], .list [.ref (heap.length + 2)],
       .list [.scalar (.bool false)], .list (List.replicate size (.scalar (.bool false)))] := by
  simp [initialHeap, distanceHeap, queueHeap, visitedHeap, List.append_assoc]

theorem initial_heap_preserves (heap : Mutable.Heap) (size : Nat) (bound distance : Int)
    (address : Nat) (object : Mutable.Object) (found : heap[address]? = some object) :
    (initialHeap heap size bound distance)[address]? = some object := by
  rw [initial_heap_layout, List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1]
  exact found

theorem initial_distances_found (heap : Mutable.Heap) (size : Nat) (bound distance : Int) :
    (initialHeap heap size bound distance)[heap.length + 1]? =
      some (.list (encode (initialDistances size bound distance))) := by
  simp [initial_heap_layout]

theorem initial_queue_found (heap : Mutable.Heap) (size : Nat) (bound distance : Int) :
    (initialHeap heap size bound distance)[heap.length + 3]? =
      some (.list [.ref (heap.length + 2)]) := by
  simp [initial_heap_layout]

theorem initial_visited_found (heap : Mutable.Heap) (size : Nat) (bound distance : Int) :
    (initialHeap heap size bound distance)[heap.length + 5]? =
      some (.list (List.replicate size (.scalar (.bool false)))) := by
  simp [initial_heap_layout]

theorem initial_queue_typed (heap : Mutable.Heap) (size : Nat) (bound distance : Int) :
    QueueEntry (initialHeap heap size bound distance) (.ref (heap.length + 2)) ∧
      priority (initialHeap heap size bound distance) (.ref (heap.length + 2)) = distance ∧
      Ordered (priority (initialHeap heap size bound distance)) [.ref (heap.length + 2)] := by
  have found : (initialHeap heap size bound distance)[heap.length + 2]? =
      some (.tuple [number distance, number 0]) := by
    simp [initial_heap_layout]
  exact ⟨⟨heap.length + 2, distance, 0, rfl, found⟩,
    priority_read _ _ distance 0 found, ProjectEulerSolutions.BinaryHeapOrder.ordered_singleton _ _⟩

/-- The two initialized Python buffers represent precisely the existing array model. -/
theorem initial_buffers_model (matrix : List (List Int)) :
    ProjectEulerSolutions.P83.SearchState.mk
      (List.replicate (ProjectEulerSolutions.DijkstraGrid.total matrix) false).toArray
      (initialDistances (ProjectEulerSolutions.DijkstraGrid.total matrix)
        (ProjectEulerStatements.P83.bigVal matrix) (ProjectEulerStatements.P83.getD2 matrix 0 0)).toArray =
      ProjectEulerSolutions.DijkstraGrid.initial matrix := by
  simp only [initialDistances, ProjectEulerSolutions.DijkstraGrid.initial,
    Array.set!_eq_setIfInBounds, ← List.toArray_replicate, List.setIfInBounds_toArray]

end solvers.«83»
