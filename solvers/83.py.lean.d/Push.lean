import solvers.«83.py.lean.d».QueueInvariant
import ProjectEulerSolutions.Lib.TriangleBufferCalls

namespace solvers.«83»
open FormalSW.PyAst HeapCalls
open ProjectEulerSolutions.TriangleBufferCalls (number)
open ProjectEulerSolutions.BinaryHeapOrder (Ordered)
set_option Elab.async false

abbrev pushCandidate : Expr := .builtinCall .heapqPush
  [.name "heap", .tuple [.name "nd", .name "v"]]

/-- The source expression allocates a fresh tuple and then pushes its reference. -/
theorem push_candidate (heap : Mutable.Heap) (env : HeapCalls.Env) (address size index : Nat)
    (infinity candidate : Int) (distance : Nat → Int) (visited : Nat → Bool) (values : List Mutable.Value)
    (found : heap[address]? = some (.list values)) (typed : ∀ value ∈ values, QueueEntry heap value)
    (ordered : Ordered (priority heap) values) (covered : CoveredQueue heap size infinity distance visited values)
    (inside : index < size) (smaller : candidate < distance index) (finite : candidate < infinity)
    (localHeap : env "heap" = some (.ref address)) (localCandidate : env "nd" = some (number candidate))
    (localIndex : env "v" = some (number index)) :
    ∃ after result, Evaluates program env pushCandidate heap (.scalar .none) after ∧
      after[address]? = some (.list result) ∧ Ordered (priority after) result ∧
      (∀ value ∈ result, QueueEntry after value) ∧
      CoveredQueue after size infinity (Function.update distance index candidate) visited result ∧
      Heapq.Reorders (heap ++ [.tuple [number candidate, number index]]) address
        (values ++ [.ref heap.length]) after := by
  let allocated := heap ++ [Mutable.Object.tuple [number candidate, number index]]
  have kept : Heapq.TupleFrame heap allocated := by
    intro target fields read
    rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp read).1]
    exact read
  have stillFound : allocated[address]? = some (.list values) := by
    rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1]
    exact found
  have itemFound : allocated[heap.length]? = some (.tuple [number candidate, number index]) := by simp [allocated]
  have newKey : priority allocated (.ref heap.length) = candidate := priority_read _ _ _ _ itemFound
  have newIndex : vertex allocated (.ref heap.length) = index := vertex_read _ _ _ _ itemFound
  have allocatedTyped : ∀ value ∈ values, QueueEntry allocated value :=
    fun value member => ((typed value member).frame kept).1
  have newTyped : QueueEntry allocated (.ref heap.length) := ⟨heap.length, candidate, index, rfl, itemFound⟩
  have allocatedOrder := queue_order_frame heap allocated values kept typed ordered
  have allocatedCoverage := CoveredQueue.frame heap allocated size infinity distance visited values typed kept covered
  obtain ⟨after, result, ran, finalFound, finalOrder, finalTyped, finalCovered, reordered⟩ :=
    CoveredQueue.push_improvement allocated address size infinity distance visited values (.ref heap.length)
      stillFound allocatedTyped newTyped allocatedOrder allocatedCoverage
      (by simpa only [newIndex] using inside) (by simpa only [newIndex, newKey] using smaller)
      (by simpa only [newKey] using finite)
  refine ⟨after, result, ?_, finalFound, finalOrder, finalTyped, ?_, reordered⟩
  · have tuple : Evaluates program env (.tuple [.name "nd", .name "v"]) heap (.ref heap.length) allocated :=
      Evaluates.tuple (.cons (Evaluates.lookup localCandidate) (.cons (Evaluates.lookup localIndex) .nil)) (by simp)
    exact Evaluates.builtin (.cons (Evaluates.lookup localHeap) (.cons tuple .nil)) ran
  · simpa only [newKey, newIndex] using finalCovered

end solvers.«83»
