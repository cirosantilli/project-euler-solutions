import solvers.«83.py.lean.d».NeighborState

namespace solvers.«83»
open FormalSW.PyAst
open ProjectEulerSolutions.RectangularMatrixCalls (MatrixRows)
open ProjectEulerSolutions.TriangleBufferCalls (encode)
set_option Elab.async false

theorem NeighborState.transport {heap after : Mutable.Heap} {matrixAddress distanceAddress queueAddress : Nat}
    {rows : MatrixRows} {width : Nat} {infinity : Int} {visited : Nat → Bool}
    {values : List Int} {queue : List Mutable.Value}
    (state : NeighborState heap matrixAddress distanceAddress queueAddress rows width infinity visited values queue)
    (matrixFound : after[matrixAddress]? = some (.list (rows.map fun pair => .ref pair.1)))
    (reads : ∀ pair ∈ rows, after[pair.1]? = some (.list (encode pair.2)))
    (distanceFound : after[distanceAddress]? = some (.list (encode values)))
    (queueFound : after[queueAddress]? = some (.list queue))
    (tuples : Heapq.TupleFrame heap after) :
    NeighborState after matrixAddress distanceAddress queueAddress rows width infinity visited values queue := by
  exact ⟨matrixFound, reads, state.matrixPrivate, state.rowsPrivate, state.separate, state.shape, state.size,
    state.bounded, distanceFound, queueFound, fun value member => ((state.typed value member).frame tuples).1,
    queue_order_frame heap after queue tuples state.typed state.ordered,
    CoveredQueue.frame heap after values.length infinity _ visited queue state.typed tuples state.covered⟩

theorem NeighborState.append {heap : Mutable.Heap} {matrixAddress distanceAddress queueAddress : Nat}
    {rows : MatrixRows} {width : Nat} {infinity : Int} {visited : Nat → Bool}
    {values : List Int} {queue : List Mutable.Value}
    (state : NeighborState heap matrixAddress distanceAddress queueAddress rows width infinity visited values queue)
    (extra : List Mutable.Object) :
    NeighborState (heap ++ extra) matrixAddress distanceAddress queueAddress rows width infinity visited values queue := by
  have kept (address : Nat) (object : Mutable.Object) (found : heap[address]? = some object) :
      (heap ++ extra)[address]? = some object := by
    rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1]
    exact found
  exact state.transport (kept _ _ state.matrixFound) (fun pair member => kept _ _ (state.reads pair member))
    (kept _ _ state.distanceFound) (kept _ _ state.queueFound) (fun address fields found => kept address _ found)

theorem NeighborState.replace_list {heap : Mutable.Heap} {matrixAddress distanceAddress queueAddress : Nat}
    {rows : MatrixRows} {width : Nat} {infinity : Int} {visited : Nat → Bool}
    {values : List Int} {queue : List Mutable.Value}
    (state : NeighborState heap matrixAddress distanceAddress queueAddress rows width infinity visited values queue)
    (address : Nat) (contents replacement : List Mutable.Value)
    (found : heap[address]? = some (.list contents))
    (notMatrix : matrixAddress ≠ address) (notRows : ∀ pair ∈ rows, pair.1 ≠ address)
    (notDistance : distanceAddress ≠ address) (notQueue : queueAddress ≠ address) :
    NeighborState (heap.set address (.list replacement)) matrixAddress distanceAddress queueAddress rows width
      infinity visited values queue := by
  apply state.transport
  · simpa only [List.getElem?_set_ne notMatrix.symm] using state.matrixFound
  · intro pair member
    simpa only [List.getElem?_set_ne (notRows pair member).symm] using state.reads pair member
  · simpa only [List.getElem?_set_ne notDistance.symm] using state.distanceFound
  · simpa only [List.getElem?_set_ne notQueue.symm] using state.queueFound
  · exact Heapq.TupleFrame.replace_list heap address contents found heap replacement (Heapq.TupleFrame.refl heap)

end solvers.«83»
