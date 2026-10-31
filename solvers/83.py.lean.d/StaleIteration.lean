import solvers.«83.py.lean.d».RuntimeLocals

namespace solvers.«83»
open FormalSW.PyAst HeapCalls
open ProjectEulerSolutions
open RectangularMatrixCalls (MatrixRows)
open TriangleBufferCalls (number)
open GridShortestPaths (Cell)
set_option Elab.async false

theorem stale_pop (heap : Mutable.Heap) (env : HeapCalls.Env)
    (matrixAddress distanceAddress visitedAddress queueAddress : Nat)
    (rows : MatrixRows) (width : Nat) (source : Cell (rows.map Prod.snd))
    (values : List Int) (flags : List Bool) (root : Mutable.Value) (rest : List Mutable.Value)
    (state : LoopState heap matrixAddress distanceAddress visitedAddress queueAddress rows width source values flags (root :: rest))
    (locals : LoopLocals env matrixAddress distanceAddress visitedAddress queueAddress rows.length width)
    (done : flags.getD (vertex heap root) false = true) :
    ∃ after nextEnv nextQueue,
      BlockContinues program env [popAssignment] heap nextEnv after ∧
      LoopState after matrixAddress distanceAddress visitedAddress queueAddress rows width source values flags nextQueue ∧
      LoopLocals nextEnv matrixAddress distanceAddress visitedAddress queueAddress rows.length width ∧
      nextEnv "u" = some (number (vertex heap root)) ∧
      nextQueue.length = rest.length ∧ after.length = heap.length ∧
      ∀ other, other ≠ queueAddress → after[other]? = heap[other]? := by
  obtain ⟨after, result, distance, index, key, indexEq, popped, found, ordered, typed, reordered⟩ :=
    pop_assignment heap env queueAddress root rest locals.queue state.neighbors.queueFound
      state.neighbors.typed state.neighbors.ordered
  subst distance
  subst index
  have finalState := state.stale_reorder heap after matrixAddress distanceAddress visitedAddress queueAddress
    rows width source values flags root rest result done reordered found ordered typed
  have frame := Heapq.Reorders.frame heap queueAddress rest after reordered
  refine ⟨after, _, result, popped, finalState, locals.pop _ _, rfl, ?_, frame⟩
  have permutation := (queueKeys_reorders heap after queueAddress (root :: rest) rest result
    state.neighbors.queueFound (fun value member => state.neighbors.typed value (List.mem_cons_of_mem _ member))
    reordered found).1
  simpa only [queueKeys, List.length_map] using permutation.length_eq

end solvers.«83»
