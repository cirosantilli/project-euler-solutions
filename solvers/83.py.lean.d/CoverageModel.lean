import solvers.«83.py.lean.d».DistanceBuffer
import ProjectEulerSolutions.Lib.LazyDijkstraQueueCongruence
import ProjectEulerSolutions.Lib.DijkstraQueueSelection

namespace solvers.«83»
open FormalSW.PyAst
open ProjectEulerSolutions
set_option Elab.async false

theorem distanceValues_array (values : List Int) (infinity : Int) (i : Nat)
    (inside : i < values.length) : distanceValues values infinity i = values.toArray[i]! := by
  rw [distanceValues_read values infinity i inside]
  simp [getElem!_pos, inside]

/-- The list and array defaults may differ outside the valid vertex domain. -/
theorem covered_buffers_iff (heap : Mutable.Heap) (infinity : Int)
    (values : List Int) (flags : List Bool) (queue : List Mutable.Value)
    (sameSize : flags.length = values.length) :
    CoveredQueue heap values.length infinity (distanceValues values infinity)
      (fun i => flags.getD i false) queue ↔
    CoveredQueue heap values.length infinity (fun i => values.toArray[i]!)
      (fun i => flags.toArray[i]!) queue := by
  have reads (i : Nat) (inside : i < values.length) : flags.getD i false = flags.toArray[i]! := by
    have valid : i < flags.length := by omega
    simp [List.getD, getElem!_pos, valid]
  constructor
  · intro covered
    exact LazyDijkstraQueue.Invariant.congr covered (distanceValues_array values infinity) reads
  · intro covered
    exact LazyDijkstraQueue.Invariant.congr covered
      (fun i inside => (distanceValues_array values infinity i inside).symm)
      (fun i inside => (reads i inside).symm)

theorem covered_model (matrix : List (List Int)) (state : P83.SearchState)
    (heap : Mutable.Heap) (values : List Int) (flags : List Bool) (queue : List Mutable.Value)
    (size : values.length = DijkstraGrid.total matrix) (sameSize : flags.length = values.length)
    (distances : values.toArray = state.distances) (visited : flags.toArray = state.visited)
    (covered : CoveredQueue heap values.length (ProjectEulerStatements.P83.bigVal matrix)
      (distanceValues values (ProjectEulerStatements.P83.bigVal matrix)) (fun i => flags.getD i false) queue) :
    DijkstraGrid.QueueInvariant matrix state (queueKeys heap queue) := by
  have arrays := (covered_buffers_iff heap _ values flags queue sameSize).mp covered
  simpa only [CoveredQueue, size, distances, visited] using arrays

/-- Updating an in-range visited-list cell realizes the queue invariant's point update. -/
theorem visited_values_set (flags : List Bool) (i : Nat) (inside : i < flags.length) :
    (fun j => (flags.set i true).getD j false) = Function.update (fun j => flags.getD j false) i true := by
  funext j
  by_cases same : j = i
  · subst j
    simp [List.getD, inside]
  · simp [List.getD, same, Ne.symm same]

theorem visited_set_model (matrix : List (List Int)) (state : P83.SearchState)
    (selected : GridShortestPaths.Cell matrix) (flags : List Bool)
    (represented : flags.toArray = state.visited) :
    (flags.set (FiniteGridIndex.index selected) true).toArray = DijkstraGrid.mark state selected := by
  simp only [DijkstraGrid.mark, ← represented, Array.set!_eq_setIfInBounds, List.setIfInBounds_toArray]

end solvers.«83»
