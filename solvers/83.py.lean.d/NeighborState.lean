import solvers.«83.py.lean.d».NeighborEvaluation
import solvers.«83.py.lean.d».NeighborArithmetic

namespace solvers.«83»
open FormalSW.PyAst HeapCalls
open ProjectEulerSolutions.RectangularMatrixCalls (MatrixRows)
open ProjectEulerSolutions.TriangleBufferCalls (number encode)
open ProjectEulerSolutions.BinaryHeapOrder (Ordered)
set_option Elab.async false

/-- The matrix is disjoint from the two buffers mutated during neighbor relaxation. -/
structure NeighborState (heap : Mutable.Heap) (matrixAddress distanceAddress queueAddress : Nat)
    (rows : MatrixRows) (width : Nat) (infinity : Int) (visited : Nat → Bool)
    (distances : List Int) (queue : List Mutable.Value) : Prop where
  matrixFound : heap[matrixAddress]? = some (.list (rows.map (fun pair => .ref pair.1)))
  reads : ∀ pair ∈ rows, heap[pair.1]? = some (.list (encode pair.2))
  matrixPrivate : matrixAddress ≠ distanceAddress ∧ matrixAddress ≠ queueAddress
  rowsPrivate : ∀ pair ∈ rows, pair.1 ≠ distanceAddress ∧ pair.1 ≠ queueAddress
  separate : distanceAddress ≠ queueAddress
  shape : ∀ pair ∈ rows, pair.2.length = width
  size : distances.length = rows.length * width
  bounded : ∀ i (inside : i < distances.length), distances[i] ≤ infinity
  distanceFound : heap[distanceAddress]? = some (.list (encode distances))
  queueFound : heap[queueAddress]? = some (.list queue)
  typed : ∀ value ∈ queue, QueueEntry heap value
  ordered : Ordered (priority heap) queue
  covered : CoveredQueue heap distances.length infinity (distanceValues distances infinity) visited queue

structure NeighborLocals (env : HeapCalls.Env) (matrixAddress distanceAddress queueAddress r c height width : Nat)
    (distance : Int) : Prop where
  matrix : env "matrix" = some (.ref matrixAddress)
  buffer : env "dist" = some (.ref distanceAddress)
  queue : env "heap" = some (.ref queueAddress)
  row : env "r" = some (number r)
  column : env "c" = some (number c)
  height : env "n" = some (number height)
  width : env "m" = some (number width)
  distance : env "d" = some (number distance)

theorem NeighborLocals.frame {env after : HeapCalls.Env}
    {matrixAddress distanceAddress queueAddress r c height width : Nat} {distance : Int}
    (locals : NeighborLocals env matrixAddress distanceAddress queueAddress r c height width distance)
    (kept : ∀ name, name ≠ "v" → name ≠ "nd" → after name = env name) :
    NeighborLocals after matrixAddress distanceAddress queueAddress r c height width distance := by
  constructor
  · rw [kept "matrix" (by decide) (by decide)]; exact locals.matrix
  · rw [kept "dist" (by decide) (by decide)]; exact locals.buffer
  · rw [kept "heap" (by decide) (by decide)]; exact locals.queue
  · rw [kept "r" (by decide) (by decide)]; exact locals.row
  · rw [kept "c" (by decide) (by decide)]; exact locals.column
  · rw [kept "n" (by decide) (by decide)]; exact locals.height
  · rw [kept "m" (by decide) (by decide)]; exact locals.width
  · rw [kept "d" (by decide) (by decide)]; exact locals.distance

def neighborUpdate (rows : MatrixRows) (width r c : Nat) (distance : Int)
    (direction : NeighborDirection) (values : List Int) : List Int :=
  if direction.enabled r c rows.length width then
    let index := direction.targetRow r * width + direction.targetColumn c
    let candidate := distance + ProjectEulerStatements.P83.getD2 (rows.map Prod.snd)
      (direction.targetRow r) (direction.targetColumn c)
    values.set index (min (values.getD index 0) candidate)
  else values

theorem neighborUpdate_length (rows : MatrixRows) (width r c : Nat) (distance : Int)
    (direction : NeighborDirection) (values : List Int) :
    (neighborUpdate rows width r c distance direction values).length = values.length := by
  unfold neighborUpdate
  split <;> simp

theorem min_update_bounded (values : List Int) (infinity candidate : Int) (index : Nat)
    (bounded : ∀ i (inside : i < values.length), values[i] ≤ infinity) :
    ∀ i (inside : i < (values.set index (min (values.getD index 0) candidate)).length),
      (values.set index (min (values.getD index 0) candidate))[i] ≤ infinity := by
  intro i inside
  have valid : i < values.length := by simpa using inside
  by_cases same : index = i
  · subst index
    rw [List.getElem_set_self]
    have old := bounded i valid
    simpa [List.getD, List.getElem?_eq_getElem valid] using
      Int.le_trans (Int.min_le_left (values.getD i 0) candidate)
        (by simpa [List.getD, List.getElem?_eq_getElem valid] using old)
  · rw [List.getElem_set_ne same]
    exact bounded i valid

theorem neighborUpdate_bounded (rows : MatrixRows) (width r c : Nat) (distance infinity : Int)
    (direction : NeighborDirection) (values : List Int)
    (bounded : ∀ i (inside : i < values.length), values[i] ≤ infinity) :
    ∀ i (inside : i < (neighborUpdate rows width r c distance direction values).length),
      (neighborUpdate rows width r c distance direction values)[i] ≤ infinity := by
  unfold neighborUpdate
  split
  · exact min_update_bounded values infinity _ _ bounded
  · exact bounded

theorem matrix_cost_read (rows : MatrixRows) (r c : Nat) (rowInside : r < rows.length)
    (columnInside : c < rows[r].2.length) :
    ProjectEulerStatements.P83.getD2 (rows.map Prod.snd) r c = rows[r].2[c] := by
  simp [ProjectEulerStatements.P83.getD2, List.getD, List.getElem?_eq_getElem rowInside,
    List.getElem?_eq_getElem columnInside]

theorem NeighborState.advance {heap after : Mutable.Heap} {matrixAddress distanceAddress queueAddress : Nat}
    {rows : MatrixRows} {width : Nat} {infinity : Int} {visited : Nat → Bool}
    {distances updated : List Int} {queue result : List Mutable.Value}
    (state : NeighborState heap matrixAddress distanceAddress queueAddress rows width infinity visited distances queue)
    (size : updated.length = distances.length)
    (bounded : ∀ i (inside : i < updated.length), updated[i] ≤ infinity)
    (distanceFound : after[distanceAddress]? = some (.list (encode updated)))
    (queueFound : after[queueAddress]? = some (.list result))
    (typed : ∀ value ∈ result, QueueEntry after value) (ordered : Ordered (priority after) result)
    (covered : CoveredQueue after distances.length infinity (distanceValues updated infinity) visited result)
    (frame : ∀ other, other < heap.length → other ≠ distanceAddress → other ≠ queueAddress → after[other]? = heap[other]?) :
    NeighborState after matrixAddress distanceAddress queueAddress rows width infinity visited updated result := by
  refine ⟨?_, ?_, state.matrixPrivate, state.rowsPrivate, state.separate, state.shape, size.trans state.size,
    bounded, distanceFound, queueFound, typed, ordered, ?_⟩
  · rw [frame matrixAddress (List.getElem?_eq_some_iff.mp state.matrixFound).1 state.matrixPrivate.1 state.matrixPrivate.2]
    exact state.matrixFound
  · intro pair member
    rw [frame pair.1 (List.getElem?_eq_some_iff.mp (state.reads pair member)).1
      (state.rowsPrivate pair member).1 (state.rowsPrivate pair member).2]
    exact state.reads pair member
  · simpa only [size] using covered

end solvers.«83»
