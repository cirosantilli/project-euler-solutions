import solvers.«83.py.lean.d».RuntimeState
import solvers.«83.py.lean.d».NeighborFrame

namespace solvers.«83»
open FormalSW.PyAst HeapCalls
open ProjectEulerSolutions
open RectangularMatrixCalls (MatrixRows)
open TriangleBufferCalls (number)
open GridShortestPaths (Cell)
set_option Elab.async false

def preparedHeap (heap : Mutable.Heap) (address : Nat) (flags : List Bool) (index width : Nat) : Mutable.Heap :=
  heap.set address (.list (encodeVisited (flags.set index true))) ++
    [.tuple [number (index / width : Nat), number (index % width : Nat)]]

def preparedEnv (env : HeapCalls.Env) (index width : Nat) : HeapCalls.Env :=
  assignNames env ["r", "c"] [number (index / width : Nat), number (index % width : Nat)]

theorem preparedHeap_length (heap : Mutable.Heap) (address : Nat) (flags : List Bool) (index width : Nat) :
    (preparedHeap heap address flags index width).length = heap.length + 1 := by simp [preparedHeap]

theorem preparedHeap_visited (heap : Mutable.Heap) (address : Nat) (flags : List Bool) (index width : Nat)
    (found : heap[address]? = some (.list (encodeVisited flags))) :
    (preparedHeap heap address flags index width)[address]? = some (.list (encodeVisited (flags.set index true))) := by
  have inside := (List.getElem?_eq_some_iff.mp found).1
  simp [preparedHeap, List.getElem?_append_left, inside]

theorem preparedHeap_frame (heap : Mutable.Heap) (address : Nat) (flags : List Bool) (index width other : Nat)
    (inside : other < heap.length) (different : other ≠ address) :
    (preparedHeap heap address flags index width)[other]? = heap[other]? := by
  simp [preparedHeap, List.getElem?_append_left, inside, different.symm]

theorem LoopState.root_inside (heap : Mutable.Heap) (matrixAddress distanceAddress visitedAddress queueAddress : Nat)
    (rows : MatrixRows) (width : Nat) (source : Cell (rows.map Prod.snd))
    (values : List Int) (flags : List Bool) (root : Mutable.Value) (rest : List Mutable.Value)
    (state : LoopState heap matrixAddress distanceAddress visitedAddress queueAddress rows width source values flags (root :: rest)) :
    vertex heap root < values.length :=
  state.neighbors.covered.valid_entry (priority heap root, vertex heap root) (by simp [queueKeys])

theorem LoopState.fresh_reorder (heap after : Mutable.Heap)
    (matrixAddress distanceAddress visitedAddress queueAddress : Nat)
    (rows : MatrixRows) (width : Nat) (source : Cell (rows.map Prod.snd))
    (values : List Int) (flags : List Bool) (root : Mutable.Value) (rest result : List Mutable.Value)
    (state : LoopState heap matrixAddress distanceAddress visitedAddress queueAddress rows width source values flags (root :: rest))
    (reordered : Heapq.Reorders heap queueAddress rest after)
    (found : after[queueAddress]? = some (.list result))
    (ordered : BinaryHeapOrder.Ordered (priority after) result) (typed : ∀ value ∈ result, QueueEntry after value) :
    NeighborState after matrixAddress distanceAddress queueAddress rows width
      (ProjectEulerStatements.P83.bigVal (rows.map Prod.snd))
      (fun i => (flags.set (vertex heap root) true).getD i false) values result := by
  have inside : vertex heap root < flags.length := by
    rw [state.flagsSize]
    exact state.root_inside heap matrixAddress distanceAddress visitedAddress queueAddress rows width source values flags root rest
  have covered := CoveredQueue.pop_mark heap after queueAddress values.length _ _ _ root rest result
    state.neighbors.queueFound state.neighbors.typed state.neighbors.covered reordered found
  rw [← visited_values_set flags (vertex heap root) inside] at covered
  have kept := (Heapq.Reorders.frame heap queueAddress rest after reordered).2
  refine ⟨?_, ?_, state.neighbors.matrixPrivate, state.neighbors.rowsPrivate, state.neighbors.separate,
    state.neighbors.shape, state.neighbors.size, state.neighbors.bounded, ?_, found, typed, ordered, covered⟩
  · rw [kept matrixAddress state.neighbors.matrixPrivate.2]
    exact state.neighbors.matrixFound
  · intro pair member
    rw [kept pair.1 (state.neighbors.rowsPrivate pair member).2]
    exact state.neighbors.reads pair member
  · rw [kept distanceAddress state.neighbors.separate]
    exact state.neighbors.distanceFound

theorem LoopState.prepare_fresh (heap after : Mutable.Heap)
    (matrixAddress distanceAddress visitedAddress queueAddress : Nat)
    (rows : MatrixRows) (width : Nat) (source : Cell (rows.map Prod.snd))
    (values : List Int) (flags : List Bool) (root : Mutable.Value) (rest result : List Mutable.Value)
    (state : LoopState heap matrixAddress distanceAddress visitedAddress queueAddress rows width source values flags (root :: rest))
    (reordered : Heapq.Reorders heap queueAddress rest after)
    (found : after[queueAddress]? = some (.list result))
    (ordered : BinaryHeapOrder.Ordered (priority after) result) (typed : ∀ value ∈ result, QueueEntry after value) :
    NeighborState (preparedHeap after visitedAddress flags (vertex heap root) width)
      matrixAddress distanceAddress queueAddress rows width (ProjectEulerStatements.P83.bigVal (rows.map Prod.snd))
      (fun i => (flags.set (vertex heap root) true).getD i false) values result := by
  have ready := state.fresh_reorder heap after matrixAddress distanceAddress visitedAddress queueAddress
    rows width source values flags root rest result reordered found ordered typed
  have visitedFound : after[visitedAddress]? = some (.list (encodeVisited flags)) := by
    rw [(Heapq.Reorders.frame heap queueAddress rest after reordered).2 visitedAddress state.visitedPrivate.2]
    exact state.visitedFound
  exact (ready.replace_list visitedAddress _ _ visitedFound state.matrixNotVisited state.rowsNotVisited
    state.visitedPrivate.1.symm state.visitedPrivate.2.symm).append _

end solvers.«83»
