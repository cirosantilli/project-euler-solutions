import solvers.«83.py.lean.d».NeighborState

namespace solvers.«83»
open FormalSW.PyAst HeapCalls
open ProjectEulerSolutions.RectangularMatrixCalls (MatrixRows)
open ProjectEulerSolutions.TriangleBufferCalls (number)
set_option Elab.async false

theorem neighbor_step (direction : NeighborDirection) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (matrixAddress distanceAddress queueAddress r c width : Nat) (rows : MatrixRows)
    (infinity distance : Int) (visited : Nat → Bool) (distances : List Int) (queue : List Mutable.Value)
    (state : NeighborState heap matrixAddress distanceAddress queueAddress rows width infinity visited distances queue)
    (locals : NeighborLocals env matrixAddress distanceAddress queueAddress r c rows.length width distance)
    (rowInside : r < rows.length) (columnInside : c < width) :
    ∃ after finalEnv result,
      BlockContinues program env [neighborBranch direction.guard direction.index direction.row direction.column] heap finalEnv after ∧
      NeighborState after matrixAddress distanceAddress queueAddress rows width infinity visited
        (neighborUpdate rows width r c distance direction distances) result ∧
      (∀ name, name ≠ "v" → name ≠ "nd" → finalEnv name = env name) ∧
      result.length ≤ queue.length + 1 ∧ heap.length ≤ after.length ∧ after.length ≤ heap.length + 1 ∧
      ∀ other, other < heap.length → other ≠ distanceAddress → other ≠ queueAddress → after[other]? = heap[other]? := by
  have tested := neighbor_guard direction heap env r c rows.length width locals.row locals.column locals.height locals.width
  by_cases active : direction.enabled r c rows.length width
  · have bounds := direction.bounds r c rows.length width rowInside columnInside active
    let targetRow := direction.targetRow r
    let targetColumn := direction.targetColumn c
    let targetIndex := targetRow * width + targetColumn
    have rowValid : targetRow < rows.length := bounds.1
    have columnValid : targetColumn < rows[targetRow].2.length := by
      rw [state.shape _ (List.getElem_mem rowValid)]
      exact bounds.2.1
    have targetValid : targetIndex < distances.length := by rw [state.size]; exact bounds.2.2
    let candidate := distance + rows[targetRow].2[targetColumn]
    let ready := setLocal (setLocal env "v" (number targetIndex)) "nd" (number candidate)
    have updated : neighborUpdate rows width r c distance direction distances =
        distances.set targetIndex (min distances[targetIndex] candidate) := by
      unfold neighborUpdate
      rw [if_pos active, matrix_cost_read rows targetRow targetColumn rowValid columnValid]
      change distances.set targetIndex (min (distances.getD targetIndex 0) candidate) = _
      simp only [List.getD, List.getElem?_eq_getElem targetValid, Option.getD_some]
    obtain ⟨after, result, ran, finalDistance, finalQueue, finalOrder, finalTyped, finalCovered,
        queueBound, heapLower, heapUpper, frame⟩ := neighbor_enabled heap env matrixAddress distanceAddress queueAddress
      targetIndex targetRow targetColumn rows distances queue infinity distance visited
      direction.guard direction.index direction.row direction.column state.matrixFound state.reads rowValid columnValid
      state.distanceFound state.queueFound state.separate targetValid (state.bounded targetIndex targetValid)
      state.typed state.ordered state.covered locals.matrix locals.distance locals.buffer locals.queue
      (by simpa only [active, decide_true] using tested)
      (neighbor_index direction heap env r c rows.length width locals.row locals.column locals.width active)
      (neighbor_row direction heap _ r c rows.length width (by simpa [setLocal] using locals.row) active)
      (neighbor_column direction heap _ r c rows.length width (by simpa [setLocal] using locals.column) active)
      direction.ordinary_row direction.ordinary_column
    rw [updated]
    refine ⟨after, ready, result, ran, ?_, ?_, queueBound, heapLower, heapUpper, frame⟩
    · apply state.advance (by simp) ?_ finalDistance finalQueue finalTyped finalOrder finalCovered frame
      simpa only [List.getD, List.getElem?_eq_getElem targetValid, Option.getD_some] using
        min_update_bounded distances infinity candidate targetIndex state.bounded
    · intro name notVertex notCandidate
      simp [ready, setLocal, notVertex, notCandidate]
  · rw [neighborUpdate, if_neg active]
    refine ⟨heap, env, queue, ?_, state, ?_, by omega, by omega, by omega, ?_⟩
    · exact neighbor_disabled heap env direction.guard direction.index direction.row direction.column
        (by simpa only [active, decide_false] using tested)
    · intros; rfl
    · intros; rfl

end solvers.«83»
