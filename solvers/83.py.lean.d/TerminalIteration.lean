import solvers.«83.py.lean.d».RuntimeLocals
import solvers.«83.py.lean.d».GoalReturn

namespace solvers.«83»
open FormalSW.PyAst HeapCalls
open ProjectEulerSolutions
open RectangularMatrixCalls (MatrixRows)
open TriangleBufferCalls (number)
open GridShortestPaths (Cell cost)
set_option Elab.async false

theorem goal_iteration (heap : Mutable.Heap) (env : HeapCalls.Env)
    (matrixAddress distanceAddress visitedAddress queueAddress : Nat)
    (rows : MatrixRows) (width : Nat) (source : Cell (rows.map Prod.snd))
    (values : List Int) (flags : List Bool) (root : Mutable.Value) (rest : List Mutable.Value)
    (state : LoopState heap matrixAddress distanceAddress visitedAddress queueAddress rows width source values flags (root :: rest))
    (locals : LoopLocals env matrixAddress distanceAddress visitedAddress queueAddress rows.length width)
    (waiting : flags.getD (vertex heap root) false = false)
    (goal : vertex heap root = rows.length * width - 1) :
    ∃ after, BlockReturns program env loopTail heap (number (priority heap root)) after ∧
      after.length = heap.length ∧
      ∀ other, other ≠ visitedAddress → other ≠ queueAddress → after[other]? = heap[other]? := by
  have inside : vertex heap root < flags.length := by
    rw [state.flagsSize]
    exact state.root_inside heap matrixAddress distanceAddress visitedAddress queueAddress rows width source values flags root rest
  have read : flags[vertex heap root] = false := by rw [← List.getD_eq_getElem flags false inside]; exact waiting
  obtain ⟨middle, result, distance, index, key, indexEq, popped, found, ordered, typed, reordered⟩ :=
    pop_assignment heap env queueAddress root rest locals.queue state.neighbors.queueFound
      state.neighbors.typed state.neighbors.ordered
  subst distance
  subst index
  have frame := Heapq.Reorders.frame heap queueAddress rest middle reordered
  have middleVisited : middle[visitedAddress]? = some (.list (encodeVisited flags)) := by
    rw [frame.2 visitedAddress state.visitedPrivate.2]
    exact state.visitedFound
  let nextEnv := assignNames env ["d", "u"] [number (priority heap root), number (vertex heap root)]
  have nextLocals := locals.pop (priority heap root) (vertex heap root)
  have returned := goal_branch middle nextEnv visitedAddress (vertex heap root) flags (priority heap root)
    middleVisited inside read nextLocals.visited rfl (by rw [goal]; exact nextLocals.goal) rfl
  refine ⟨middle.set visitedAddress (.list (encodeVisited (flags.set (vertex heap root) true))), ?_,
    by simpa only [List.length_set] using frame.1, ?_⟩
  · apply loop_enter heap _ env queueAddress root rest _ locals.queue state.neighbors.queueFound
    exact popped.returns returned
  · intro other notVisited notQueue
    rw [List.getElem?_set_ne notVisited.symm, frame.2 other notQueue]

theorem LoopState.goal_priority (heap : Mutable.Heap)
    (matrixAddress distanceAddress visitedAddress queueAddress : Nat)
    (rows : MatrixRows) (width : Nat) (source target : Cell (rows.map Prod.snd))
    (values : List Int) (flags : List Bool) (root : Mutable.Value) (rest : List Mutable.Value)
    (nonnegative : ∀ row ∈ rows.map Prod.snd, ∀ value ∈ row, 0 ≤ value)
    (state : LoopState heap matrixAddress distanceAddress visitedAddress queueAddress rows width source values flags (root :: rest))
    (waiting : flags.getD (vertex heap root) false = false)
    (targetIndex : FiniteGridIndex.index target = vertex heap root) (answer : Int)
    (optimal : ∃ shortest : (DijkstraGrid.grid (rows.map Prod.snd)).Walk target source,
      (shortest.support.map (cost (rows.map Prod.snd))).sum = answer ∧
      ∀ walk : (DijkstraGrid.grid (rows.map Prod.snd)).Walk target source,
        answer ≤ (walk.support.map (cost (rows.map Prod.snd))).sum) :
    priority heap root = answer := by
  have size : values.length = DijkstraGrid.total (rows.map Prod.snd) := by
    simpa only [DijkstraGrid.total, List.length_map, state.matrixWidth] using state.neighbors.size
  have covered := covered_model (rows.map Prod.snd) (bufferModel flags values) heap values flags (root :: rest)
    size state.flagsSize rfl rfl state.neighbors.covered
  obtain ⟨selected, readIndex, path, weight, least⟩ := queue_root_minimum (rows.map Prod.snd) nonnegative source
    (bufferModel flags values) state.invariant heap root rest covered state.neighbors.ordered
    (by simpa [bufferModel, List.getD] using waiting)
  have same : selected = target := FiniteGridIndex.index_injective _ _ (readIndex.trans targetIndex.symm)
  subst selected
  obtain ⟨shortest, value, minimum⟩ := optimal
  have lower := minimum path
  have upper := least shortest
  omega

end solvers.«83»
