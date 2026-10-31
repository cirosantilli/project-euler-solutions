import solvers.«83.py.lean.d».NeighborGrid
import solvers.«83.py.lean.d».CoverageModel
import solvers.«83.py.lean.d».FreshControl

namespace solvers.«83»
open FormalSW.PyAst HeapCalls
open ProjectEulerSolutions
open RectangularMatrixCalls (MatrixRows)
open TriangleBufferCalls (number)
open GridShortestPaths (Cell)
set_option Elab.async false

def bufferModel (flags : List Bool) (values : List Int) : P83.SearchState :=
  ⟨flags.toArray, values.toArray⟩

/-- The mutable runtime buffers, queue coverage, and shortest-path invariant at a loop boundary. -/
structure LoopState (heap : Mutable.Heap) (matrixAddress distanceAddress visitedAddress queueAddress : Nat)
    (rows : MatrixRows) (width : Nat) (source : Cell (rows.map Prod.snd))
    (values : List Int) (flags : List Bool) (queue : List Mutable.Value) : Prop where
  neighbors : NeighborState heap matrixAddress distanceAddress queueAddress rows width
    (ProjectEulerStatements.P83.bigVal (rows.map Prod.snd)) (fun i => flags.getD i false) values queue
  visitedFound : heap[visitedAddress]? = some (.list (encodeVisited flags))
  visitedPrivate : visitedAddress ≠ distanceAddress ∧ visitedAddress ≠ queueAddress
  matrixNotVisited : matrixAddress ≠ visitedAddress
  rowsNotVisited : ∀ pair ∈ rows, pair.1 ≠ visitedAddress
  flagsSize : flags.length = values.length
  matrixWidth : ((rows.map Prod.snd).headD []).length = width
  invariant : DijkstraGrid.Invariant (rows.map Prod.snd) source (bufferModel flags values)

structure LoopLocals (env : HeapCalls.Env) (matrixAddress distanceAddress visitedAddress queueAddress heightValue widthValue : Nat) : Prop where
  matrix : env "matrix" = some (.ref matrixAddress)
  buffer : env "dist" = some (.ref distanceAddress)
  visited : env "visited" = some (.ref visitedAddress)
  queue : env "heap" = some (.ref queueAddress)
  height : env "n" = some (number heightValue)
  width : env "m" = some (number widthValue)
  goal : env "goal" = some (number (heightValue * widthValue - 1 : Nat))

theorem LoopLocals.frame {env after : HeapCalls.Env}
    {matrixAddress distanceAddress visitedAddress queueAddress height width : Nat}
    (locals : LoopLocals env matrixAddress distanceAddress visitedAddress queueAddress height width)
    (kept : ∀ name, name ≠ "d" → name ≠ "u" → name ≠ "r" → name ≠ "c" → name ≠ "v" → name ≠ "nd" →
      after name = env name) :
    LoopLocals after matrixAddress distanceAddress visitedAddress queueAddress height width := by
  constructor
  · rw [kept "matrix" (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]; exact locals.matrix
  · rw [kept "dist" (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]; exact locals.buffer
  · rw [kept "visited" (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]; exact locals.visited
  · rw [kept "heap" (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]; exact locals.queue
  · rw [kept "n" (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]; exact locals.height
  · rw [kept "m" (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]; exact locals.width
  · rw [kept "goal" (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]; exact locals.goal

theorem LoopState.selected (heap : Mutable.Heap) (matrixAddress distanceAddress visitedAddress queueAddress : Nat)
    (rows : MatrixRows) (width : Nat) (source : Cell (rows.map Prod.snd))
    (values : List Int) (flags : List Bool) (root : Mutable.Value) (rest : List Mutable.Value)
    (state : LoopState heap matrixAddress distanceAddress visitedAddress queueAddress rows width source values flags (root :: rest))
    (waiting : flags.getD (vertex heap root) false = false) :
    ∃ selected : Cell (rows.map Prod.snd),
      FiniteGridIndex.index selected = vertex heap root ∧
      DijkstraGrid.distance (bufferModel flags values) selected = priority heap root ∧
      DijkstraGrid.visited (bufferModel flags values) selected = false ∧
      DijkstraGrid.distance (bufferModel flags values) selected < ProjectEulerStatements.P83.bigVal (rows.map Prod.snd) ∧
      ∀ v : Cell (rows.map Prod.snd), DijkstraGrid.visited (bufferModel flags values) v = false →
        DijkstraGrid.distance (bufferModel flags values) selected ≤ DijkstraGrid.distance (bufferModel flags values) v := by
  have size : values.length = DijkstraGrid.total (rows.map Prod.snd) := by
    simpa only [DijkstraGrid.total, List.length_map, state.matrixWidth] using state.neighbors.size
  have covered := covered_model (rows.map Prod.snd) (bufferModel flags values) heap values flags (root :: rest)
    size state.flagsSize rfl rfl state.neighbors.covered
  apply DijkstraGrid.queue_selected (rows.map Prod.snd) (bufferModel flags values)
    (priority heap root, vertex heap root) (queueKeys heap rest) covered
  · intro entry member
    obtain ⟨value, present, rfl⟩ := (List.mem_map
      (f := fun value => (priority heap value, vertex heap value)) (l := root :: rest)).mp member
    exact BinaryHeapOrder.root_le_member (priority heap) root rest state.neighbors.ordered value present
  · simpa [bufferModel, List.getD] using waiting

theorem LoopState.stale_reorder (heap after : Mutable.Heap)
    (matrixAddress distanceAddress visitedAddress queueAddress : Nat)
    (rows : MatrixRows) (width : Nat) (source : Cell (rows.map Prod.snd))
    (values : List Int) (flags : List Bool) (root : Mutable.Value) (rest result : List Mutable.Value)
    (state : LoopState heap matrixAddress distanceAddress visitedAddress queueAddress rows width source values flags (root :: rest))
    (done : flags.getD (vertex heap root) false = true)
    (reordered : Heapq.Reorders heap queueAddress rest after)
    (found : after[queueAddress]? = some (.list result))
    (ordered : BinaryHeapOrder.Ordered (priority after) result) (typed : ∀ value ∈ result, QueueEntry after value) :
    LoopState after matrixAddress distanceAddress visitedAddress queueAddress rows width source values flags result := by
  have kept := (Heapq.Reorders.frame heap queueAddress rest after reordered).2
  have covered := CoveredQueue.drop_visited heap after queueAddress values.length _ _ _ root rest result
    state.neighbors.queueFound state.neighbors.typed state.neighbors.covered done reordered found
  refine ⟨state.neighbors.advance rfl state.neighbors.bounded ?_ found typed ordered covered
      (fun other _ _ notQueue => kept other notQueue), ?_, state.visitedPrivate,
      state.matrixNotVisited, state.rowsNotVisited, state.flagsSize, state.matrixWidth, state.invariant⟩
  · rw [kept distanceAddress state.neighbors.separate]
    exact state.neighbors.distanceFound
  · rw [kept visitedAddress state.visitedPrivate.2]
    exact state.visitedFound

end solvers.«83»
