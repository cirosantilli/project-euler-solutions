import solvers.«83.py.lean.d».NeighborStep

namespace solvers.«83»
open FormalSW.PyAst HeapCalls
open ProjectEulerSolutions.RectangularMatrixCalls (MatrixRows)
set_option Elab.async false

theorem neighbor_fold (directions : List NeighborDirection) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (matrixAddress distanceAddress queueAddress r c width : Nat) (rows : MatrixRows)
    (infinity distance : Int) (visited : Nat → Bool) (distances : List Int) (queue : List Mutable.Value)
    (state : NeighborState heap matrixAddress distanceAddress queueAddress rows width infinity visited distances queue)
    (locals : NeighborLocals env matrixAddress distanceAddress queueAddress r c rows.length width distance)
    (rowInside : r < rows.length) (columnInside : c < width) :
    ∃ after finalEnv result,
      BlockContinues program env
        (directions.map fun direction => neighborBranch direction.guard direction.index direction.row direction.column)
        heap finalEnv after ∧
      NeighborState after matrixAddress distanceAddress queueAddress rows width infinity visited
        (directions.foldl (fun values direction => neighborUpdate rows width r c distance direction values) distances) result ∧
      (∀ name, name ≠ "v" → name ≠ "nd" → finalEnv name = env name) ∧
      result.length ≤ queue.length + directions.length ∧ heap.length ≤ after.length ∧
      after.length ≤ heap.length + directions.length ∧
      ∀ other, other < heap.length → other ≠ distanceAddress → other ≠ queueAddress → after[other]? = heap[other]? := by
  induction directions generalizing heap env distances queue with
  | nil =>
      exact ⟨heap, env, queue, BlockContinues.nil, state, by intros; rfl, by simp, by omega, by simp, by intros; rfl⟩
  | cons direction rest ih =>
      obtain ⟨middle, nextEnv, nextQueue, first, nextState, nextLocals, firstCount, firstLower, firstUpper, firstFrame⟩ :=
        neighbor_step direction heap env matrixAddress distanceAddress queueAddress r c width rows infinity distance
          visited distances queue state locals rowInside columnInside
      obtain ⟨after, finalEnv, result, remaining, finalState, finalLocals, restCount, restLower, restUpper, restFrame⟩ :=
        ih middle nextEnv _ nextQueue nextState (locals.frame nextLocals)
      refine ⟨after, finalEnv, result, first.append remaining, finalState, ?_, ?_, by omega, ?_, ?_⟩
      · intro name notVertex notCandidate
        rw [finalLocals name notVertex notCandidate, nextLocals name notVertex notCandidate]
      · simp only [List.length_cons]
        omega
      · simp only [List.length_cons]
        omega
      · intro other inside notDistance notQueue
        rw [restFrame other (by omega) notDistance notQueue, firstFrame other inside notDistance notQueue]

def neighborsDistances (rows : MatrixRows) (width r c : Nat) (distance : Int) (values : List Int) : List Int :=
  ([.up, .down, .left, .right] : List NeighborDirection).foldl
    (fun values direction => neighborUpdate rows width r c distance direction values) values

/-- All four actual branches preserve the matrix, queue invariant, and unrelated objects. -/
theorem neighbors_result (heap : Mutable.Heap) (env : HeapCalls.Env)
    (matrixAddress distanceAddress queueAddress r c width : Nat) (rows : MatrixRows)
    (infinity distance : Int) (visited : Nat → Bool) (distances : List Int) (queue : List Mutable.Value)
    (state : NeighborState heap matrixAddress distanceAddress queueAddress rows width infinity visited distances queue)
    (locals : NeighborLocals env matrixAddress distanceAddress queueAddress r c rows.length width distance)
    (rowInside : r < rows.length) (columnInside : c < width) :
    ∃ after finalEnv result,
      BlockContinues program env neighborsCode heap finalEnv after ∧
      NeighborState after matrixAddress distanceAddress queueAddress rows width infinity visited
        (neighborsDistances rows width r c distance distances) result ∧
      (∀ name, name ≠ "v" → name ≠ "nd" → finalEnv name = env name) ∧
      result.length ≤ queue.length + 4 ∧ heap.length ≤ after.length ∧ after.length ≤ heap.length + 4 ∧
      ∀ other, other < heap.length → other ≠ distanceAddress → other ≠ queueAddress → after[other]? = heap[other]? := by
  rw [directions_source]
  exact neighbor_fold [.up, .down, .left, .right] heap env matrixAddress distanceAddress queueAddress r c width rows
    infinity distance visited distances queue state locals rowInside columnInside

end solvers.«83»
