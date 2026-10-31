import solvers.«83.py.lean.d».DistanceBuffer

namespace solvers.«83»
open FormalSW.PyAst HeapCalls
open ProjectEulerSolutions.TriangleBufferCalls (number encode)
open ProjectEulerSolutions.BinaryHeapOrder (Ordered)
set_option Elab.async false

/-- Exact buffer effects and queue coverage of the shared conditional relaxation block. -/
theorem relaxation_result (heap : Mutable.Heap) (env : HeapCalls.Env) (distanceAddress queueAddress index : Nat)
    (distances : List Int) (queue : List Mutable.Value) (infinity candidate : Int) (visited : Nat → Bool)
    (distanceFound : heap[distanceAddress]? = some (.list (encode distances)))
    (queueFound : heap[queueAddress]? = some (.list queue)) (separate : distanceAddress ≠ queueAddress)
    (inside : index < distances.length) (bounded : distances[index] ≤ infinity)
    (typed : ∀ value ∈ queue, QueueEntry heap value) (ordered : Ordered (priority heap) queue)
    (covered : CoveredQueue heap distances.length infinity (distanceValues distances infinity) visited queue)
    (localDistance : env "dist" = some (.ref distanceAddress)) (localQueue : env "heap" = some (.ref queueAddress))
    (localIndex : env "v" = some (number index)) (localCandidate : env "nd" = some (number candidate)) :
    let updated := distances.set index (min distances[index] candidate)
    ∃ after result, BlockContinues program env relaxationCode heap env after ∧
      after[distanceAddress]? = some (.list (encode updated)) ∧
      after[queueAddress]? = some (.list result) ∧ Ordered (priority after) result ∧
      (∀ value ∈ result, QueueEntry after value) ∧
      CoveredQueue after distances.length infinity (distanceValues updated infinity) visited result ∧
      result.length ≤ queue.length + 1 ∧ heap.length ≤ after.length ∧ after.length ≤ heap.length + 1 ∧
      ∀ other, other < heap.length → other ≠ distanceAddress → other ≠ queueAddress → after[other]? = heap[other]? := by
  dsimp only
  have tested := distance_test heap env distanceAddress index distances candidate distanceFound inside
    localDistance localIndex localCandidate
  by_cases smaller : candidate < distances[index]
  · rw [Int.min_eq_right (by omega : candidate ≤ distances[index])]
    let written := heap.set distanceAddress (.list (encode (distances.set index candidate)))
    have distanceValid := (List.getElem?_eq_some_iff.mp distanceFound).1
    have queueValid := (List.getElem?_eq_some_iff.mp queueFound).1
    have kept : Heapq.TupleFrame heap written :=
      (Heapq.TupleFrame.replace_list heap distanceAddress (encode distances) distanceFound)
        heap _ (Heapq.TupleFrame.refl heap)
    have writtenQueue : written[queueAddress]? = some (.list queue) := by simp [written, separate, queueFound]
    have writtenDistance : written[distanceAddress]? = some (.list (encode (distances.set index candidate))) := by
      simp [written, distanceValid]
    have writtenTyped : ∀ value ∈ queue, QueueEntry written value := fun value member => ((typed value member).frame kept).1
    have writtenOrder := queue_order_frame heap written queue kept typed ordered
    have writtenCoverage := CoveredQueue.frame heap written distances.length infinity (distanceValues distances infinity)
      visited queue typed kept covered
    obtain ⟨after, result, pushed, finalQueue, finalOrder, finalTyped, finalCoverage, reordered⟩ :=
      push_candidate written env queueAddress distances.length index infinity candidate
        (distanceValues distances infinity) visited queue writtenQueue writtenTyped writtenOrder writtenCoverage inside
        (by simpa only [distanceValues_read distances infinity index inside] using smaller) (by omega)
        localQueue localCandidate localIndex
    let allocated := written ++ [Mutable.Object.tuple [number candidate, number index]]
    have frame := Heapq.Reorders.frame allocated queueAddress (queue ++ [.ref written.length]) after reordered
    have extendedQueue : allocated[queueAddress]? = some (.list queue) := by
      rw [List.getElem?_append_left (by simpa [written] using queueValid)]
      exact writtenQueue
    obtain ⟨changed, changedFound, permutation⟩ := Heapq.Reorders.contents allocated queueAddress
      (queue ++ [.ref written.length]) after (List.getElem?_eq_some_iff.mp extendedQueue).1 reordered
    have same : changed = result := by simpa only [finalQueue, Option.some.injEq, Mutable.Object.list.injEq] using changedFound.symm
    subst changed
    refine ⟨after, result, ?_, ?_, finalQueue, finalOrder, finalTyped, ?_, ?_, ?_, ?_, ?_⟩
    · apply BlockContinues.branch (flag := true) (by simpa only [smaller, decide_true] using tested) rfl
      exact (distance_write heap env distanceAddress index distances candidate distanceFound inside
        localDistance localIndex localCandidate).append (BlockContinues.discard pushed)
    · rw [frame.2 distanceAddress separate, List.getElem?_append_left (by simpa [written] using distanceValid)]
      exact writtenDistance
    · rw [distanceValues_set distances infinity candidate index inside]
      exact finalCoverage
    · have length := permutation.length_eq
      simp only [List.length_append, List.length_singleton] at length
      omega
    · rw [frame.1]
      simp [allocated, written]
    · rw [frame.1]
      simp [allocated, written]
    · intro other valid notDistance notQueue
      rw [frame.2 other notQueue, List.getElem?_append_left (by simpa [written] using valid)]
      exact List.getElem?_set_ne (Ne.symm notDistance)
  · have unchanged : distances.set index (min distances[index] candidate) = distances := by
      rw [Int.min_eq_left (by omega)]
      exact List.set_getElem_self inside
    rw [unchanged]
    refine ⟨heap, queue, ?_, distanceFound, queueFound, ordered, typed, covered, by omega, by omega, by omega, ?_⟩
    · exact BlockContinues.branch (flag := false) (by simpa only [smaller, decide_false] using tested) rfl BlockContinues.nil
    · intros
      rfl

end solvers.«83»
