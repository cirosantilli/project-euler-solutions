import solvers.«83.py.lean.d».RuntimeLocals

namespace solvers.«83»
open FormalSW.PyAst HeapCalls
open ProjectEulerSolutions
open RectangularMatrixCalls (MatrixRows)
open TriangleBufferCalls (number)
open GridShortestPaths (Cell)
set_option Elab.async false

/-- A fresh non-goal iteration executes the complete Python body and restores the combined loop invariant. -/
theorem fresh_iteration (heap : Mutable.Heap) (env : HeapCalls.Env)
    (matrixAddress distanceAddress visitedAddress queueAddress : Nat)
    (rows : MatrixRows) (width : Nat) (source : Cell (rows.map Prod.snd))
    (values : List Int) (flags : List Bool) (root : Mutable.Value) (rest : List Mutable.Value)
    (nonnegative : ∀ row ∈ rows.map Prod.snd, ∀ value ∈ row, 0 ≤ value)
    (state : LoopState heap matrixAddress distanceAddress visitedAddress queueAddress rows width source values flags (root :: rest))
    (locals : LoopLocals env matrixAddress distanceAddress visitedAddress queueAddress rows.length width)
    (waiting : flags.getD (vertex heap root) false = false)
    (notGoal : vertex heap root ≠ rows.length * width - 1) :
    ∃ after finalEnv nextValues nextQueue,
      BlockContinues program env loopBody heap finalEnv after ∧
      LoopState after matrixAddress distanceAddress visitedAddress queueAddress rows width source
        nextValues (flags.set (vertex heap root) true) nextQueue ∧
      LoopLocals finalEnv matrixAddress distanceAddress visitedAddress queueAddress rows.length width ∧
      nextQueue.length ≤ rest.length + 4 ∧ heap.length ≤ after.length ∧ after.length ≤ heap.length + 5 ∧
      ∀ other, other < heap.length → other ≠ distanceAddress → other ≠ visitedAddress → other ≠ queueAddress →
        after[other]? = heap[other]? := by
  have inside := state.root_inside heap matrixAddress distanceAddress visitedAddress queueAddress rows width source values flags root rest
  have flagInside : vertex heap root < flags.length := by rw [state.flagsSize]; exact inside
  have waitingRead : flags[vertex heap root] = false := by
    rw [← List.getD_eq_getElem flags false flagInside]
    exact waiting
  have positive : 0 < width := by have size := state.neighbors.size; nlinarith
  obtain ⟨selected, readIndex, value, pending, finite, least⟩ :=
    state.selected heap matrixAddress distanceAddress visitedAddress queueAddress rows width source values flags root rest waiting
  have rowIndex : vertex heap root / width = selected.1.val := by
    rw [← readIndex, ← state.matrixWidth]
    exact FiniteGridIndex.index_div selected
  have columnIndex : vertex heap root % width = selected.2.val := by
    rw [← readIndex, ← state.matrixWidth]
    exact FiniteGridIndex.index_mod selected
  have rowInside : selected.1.val < rows.length := by simpa only [List.length_map] using selected.1.isLt
  have columnInside : selected.2.val < width := by simpa only [state.matrixWidth] using selected.2.isLt
  obtain ⟨middle, result, distance, index, key, indexEq, popped, found, ordered, typed, reordered⟩ :=
    pop_assignment heap env queueAddress root rest locals.queue state.neighbors.queueFound
      state.neighbors.typed state.neighbors.ordered
  subst distance
  subst index
  have middleFrame := Heapq.Reorders.frame heap queueAddress rest middle reordered
  have middleVisited : middle[visitedAddress]? = some (.list (encodeVisited flags)) := by
    rw [middleFrame.2 visitedAddress state.visitedPrivate.2]
    exact state.visitedFound
  have resultLength : result.length = rest.length := by
    have permutation := (queueKeys_reorders heap middle queueAddress (root :: rest) rest result
      state.neighbors.queueFound (fun item member => state.neighbors.typed item (List.mem_cons_of_mem _ member))
      reordered found).1
    simpa only [queueKeys, List.length_map] using permutation.length_eq
  let poppedEnv := assignNames env ["d", "u"] [number (priority heap root), number (vertex heap root)]
  let readyEnv := preparedEnv poppedEnv (vertex heap root) width
  let readyHeap := preparedHeap middle visitedAddress flags (vertex heap root) width
  have poppedLocals := locals.pop (priority heap root) (vertex heap root)
  have readyLocals : NeighborLocals readyEnv matrixAddress distanceAddress queueAddress
      selected.1.val selected.2.val rows.length width (priority heap root) := by
    simpa only [rowIndex, columnIndex] using poppedLocals.neighbor_locals (vertex heap root) (priority heap root) rfl
  have readyState := state.prepare_fresh heap middle matrixAddress distanceAddress visitedAddress queueAddress
    rows width source values flags root rest result reordered found ordered typed
  have readyVisited := preparedHeap_visited middle visitedAddress flags (vertex heap root) width middleVisited
  have readyLength : readyHeap.length = heap.length + 1 := by
    rw [preparedHeap_length, middleFrame.1]
  obtain ⟨after, finalEnv, nextQueue, relaxed, nextState, localFrame, queueCount, heapLower, heapUpper, heapFrame⟩ :=
    neighbors_result readyHeap readyEnv matrixAddress distanceAddress queueAddress selected.1.val selected.2.val width
      rows (ProjectEulerStatements.P83.bigVal (rows.map Prod.snd)) (priority heap root)
      (fun i => (flags.set (vertex heap root) true).getD i false) values result readyState readyLocals rowInside columnInside
  let nextValues := neighborsDistances rows width selected.1.val selected.2.val (priority heap root) values
  have nextVisited : after[visitedAddress]? = some (.list (encodeVisited (flags.set (vertex heap root) true))) := by
    rw [heapFrame visitedAddress (List.getElem?_eq_some_iff.mp readyVisited).1 state.visitedPrivate.1 state.visitedPrivate.2]
    exact readyVisited
  have invariant := neighbors_grid_invariant rows width state.matrixWidth nonnegative source selected
    (bufferModel flags values) state.invariant pending finite least values rfl
  rw [← visited_set_model (rows.map Prod.snd) (bufferModel flags values) selected flags rfl, readIndex, value] at invariant
  have nextFlagsSize : (flags.set (vertex heap root) true).length = nextValues.length := by
    have nextSize := nextState.size
    have oldSize := state.neighbors.size
    rw [List.length_set, state.flagsSize]
    exact oldSize.trans nextSize.symm
  have finalLocals : LoopLocals finalEnv matrixAddress distanceAddress visitedAddress queueAddress rows.length width :=
    (poppedLocals.coordinates (vertex heap root)).frame
      (fun name _ _ _ _ notVertex notCandidate => localFrame name notVertex notCandidate)
  refine ⟨after, finalEnv, nextValues, nextQueue, ?_,
    ⟨nextState, nextVisited, state.visitedPrivate, state.matrixNotVisited, state.rowsNotVisited,
      nextFlagsSize, state.matrixWidth, invariant⟩,
    finalLocals, by omega, by omega, by omega, ?_⟩
  · have prefixRan := fresh_prefix middle poppedEnv visitedAddress (vertex heap root) (rows.length * width - 1) width
      flags middleVisited flagInside waitingRead notGoal positive
      poppedLocals.visited rfl poppedLocals.goal poppedLocals.width
    exact popped.append (prefixRan.append relaxed)
  · intro other oldInside notDistance notVisited notQueue
    rw [heapFrame other (by omega) notDistance notQueue,
      preparedHeap_frame middle visitedAddress flags (vertex heap root) width other
        (by omega) notVisited, middleFrame.2 other notQueue]

end solvers.«83»
