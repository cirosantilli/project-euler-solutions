import solvers.«83.py.lean.d».RuntimeState
import ProjectEulerSolutions.Lib.DijkstraGridProgress
import ProjectEulerSolutions.Lib.VisitedListProgress

namespace solvers.«83»
open FormalSW.PyAst
open ProjectEulerSolutions
open RectangularMatrixCalls (MatrixRows)
open GridShortestPaths (Cell)
set_option Elab.async false

def loopMeasure (flags : List Bool) (queue : List Mutable.Value) : Nat :=
  5 * flags.count false + queue.length

theorem fresh_measure (flags : List Bool) (index : Nat) (inside : index < flags.length)
    (waiting : flags.getD index false = false) (root : Mutable.Value) (rest nextQueue : List Mutable.Value)
    (count : nextQueue.length ≤ rest.length + 4) :
    loopMeasure (flags.set index true) nextQueue < loopMeasure flags (root :: rest) := by
  have read : flags[index] = false := by rw [← List.getD_eq_getElem flags false inside]; exact waiting
  exact LazyDijkstraQueue.fresh_potential_decreases 4 _ _ rest.length nextQueue.length
    (VisitedListProgress.mark_decreases flags index inside read) count

theorem stale_measure (flags : List Bool) (root : Mutable.Value) (rest nextQueue : List Mutable.Value)
    (count : nextQueue.length = rest.length) :
    loopMeasure flags nextQueue < loopMeasure flags (root :: rest) := by
  simp only [loopMeasure, List.length_cons, count]
  omega

/-- An unvisited reachable cell guarantees a finite queued frontier key. -/
theorem LoopState.queue_nonempty (heap : Mutable.Heap)
    (matrixAddress distanceAddress visitedAddress queueAddress : Nat)
    (rows : MatrixRows) (width : Nat) (source target : Cell (rows.map Prod.snd))
    (values : List Int) (flags : List Bool) (queue : List Mutable.Value)
    (nonnegative : ∀ row ∈ rows.map Prod.snd, ∀ value ∈ row, 0 ≤ value)
    (reachable : ∀ v : Cell (rows.map Prod.snd), Nonempty ((DijkstraGrid.grid (rows.map Prod.snd)).Walk v source))
    (state : LoopState heap matrixAddress distanceAddress visitedAddress queueAddress rows width source values flags queue)
    (waiting : flags.getD (FiniteGridIndex.index target) false = false) : queue ≠ [] := by
  have shape : ∀ row ∈ rows.map Prod.snd, row.length = ((rows.map Prod.snd).headD []).length := by
    intro row member
    obtain ⟨pair, present, rfl⟩ := List.mem_map.mp member
    exact (state.neighbors.shape pair present).trans state.matrixWidth.symm
  have pending : DijkstraGrid.visited (bufferModel flags values) target = false := by
    simpa [DijkstraGrid.visited, bufferModel, List.getD] using waiting
  obtain ⟨i, inside, unvisited, finite⟩ := DijkstraGrid.finite_available (rows.map Prod.snd) shape nonnegative
    source reachable (bufferModel flags values) state.invariant ⟨target, pending⟩
  have size : values.length = DijkstraGrid.total (rows.map Prod.snd) := by
    simpa only [DijkstraGrid.total, List.length_map, state.matrixWidth] using state.neighbors.size
  have covered := covered_model (rows.map Prod.snd) (bufferModel flags values) heap values flags queue
    size state.flagsSize rfl rfl state.neighbors.covered
  have nonempty := LazyDijkstraQueue.Invariant.nonempty covered i inside unvisited finite
  intro empty
  exact nonempty (by simp [queueKeys, empty])

end solvers.«83»
