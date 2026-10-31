import solvers.«83.py.lean.d».QueueInvariant
import solvers.«83.py.lean.d».Entry

namespace solvers.«83»
open FormalSW.PyAst HeapCalls
open ProjectEulerSolutions.TriangleBufferCalls (number)
open ProjectEulerSolutions.BinaryHeapOrder (Ordered)
set_option Elab.async false

abbrev popAssignment : Stmt := .assign [.tuple [.name "d", .name "u"]]
  (.builtinCall .heapqPop [.name "heap"])

theorem pop_source : ∃ body otherwise following, loopTail = .whileDo (.name "heap") body otherwise :: following ∧
    body.head? = some popAssignment := by
  exact ⟨_, _, _, rfl, rfl⟩

/-- The actual pop and unpack bind the original tuple fields and preserve every other heap object. -/
theorem pop_assignment (heap : Mutable.Heap) (env : HeapCalls.Env) (address : Nat)
    (root : Mutable.Value) (rest : List Mutable.Value)
    (localHeap : env "heap" = some (.ref address))
    (found : heap[address]? = some (.list (root :: rest)))
    (typed : ∀ value ∈ root :: rest, QueueEntry heap value)
    (ordered : Ordered (priority heap) (root :: rest)) :
    ∃ after result distance index,
      priority heap root = distance ∧ vertex heap root = index ∧
      BlockContinues program env [popAssignment] heap
        (assignNames env ["d", "u"] [number distance, number index]) after ∧
      after[address]? = some (.list result) ∧ Ordered (priority after) result ∧
      (∀ value ∈ result, QueueEntry after value) ∧ Heapq.Reorders heap address rest after := by
  obtain ⟨tupleAddress, distance, index, rfl, readTuple⟩ := typed root (by simp)
  obtain ⟨after, result, ran, finalFound, finalOrder, reordered, _⟩ :=
    queue_pop_ordered heap address (.ref tupleAddress) rest found typed ordered
  have kept := (Heapq.TupleFrame.replace_list heap address (.ref tupleAddress :: rest) found).reorders
    (Heapq.TupleFrame.refl heap) reordered
  have retained := kept tupleAddress [distance, index] readTuple
  refine ⟨after, result, distance, index, priority_read heap tupleAddress distance index readTuple,
    vertex_read heap tupleAddress distance index readTuple, ?_, finalFound, finalOrder, ?_, reordered⟩
  · have popped : Evaluates program env (.builtinCall .heapqPop [.name "heap"]) heap (.ref tupleAddress) after :=
      Evaluates.builtin (.cons (Evaluates.lookup localHeap) .nil) ran
    apply BlockContinues.unpack (names := ["d", "u"]) rfl popped
      (items := [number distance, number index]) ?_ rfl
    simp [Mutable.sequence, retained]
  · exact (queueKeys_reorders heap after address (.ref tupleAddress :: rest) rest result found
      (fun value member => typed value (List.mem_cons_of_mem _ member)) reordered finalFound).2

end solvers.«83»
