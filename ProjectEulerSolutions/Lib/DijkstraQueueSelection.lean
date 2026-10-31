import ProjectEulerSolutions.Lib.LazyDijkstraQueue
import ProjectEulerSolutions.Lib.DijkstraGridInvariant

/-! Priority-queue selection supplies the minimum-distance hypothesis of the grid proof. -/
namespace ProjectEulerSolutions.DijkstraGrid
open ProjectEulerSolutions.P83
open ProjectEulerStatements.P83 (bigVal)
open FiniteGridIndex (index)
open GridShortestPaths (Cell cost)

def QueueInvariant (matrix : List (List Int)) (state : SearchState) (queue : List (Int × Nat)) : Prop :=
  LazyDijkstraQueue.Invariant (fun vertex => vertex < total matrix) (bigVal matrix)
    (fun vertex => state.distances[vertex]!) (fun vertex => state.visited[vertex]!) queue

theorem queue_selected (matrix : List (List Int)) (state : SearchState) (root : Int × Nat)
    (rest : List (Int × Nat)) (queue : QueueInvariant matrix state (root :: rest))
    (minimum : ∀ entry ∈ root :: rest, root.1 ≤ entry.1) (waiting : state.visited[root.2]! = false) :
    ∃ selected : Cell matrix, index selected = root.2 ∧ distance state selected = root.1 ∧
      visited state selected = false ∧ distance state selected < bigVal matrix ∧
      ∀ v : Cell matrix, visited state v = false → distance state selected ≤ distance state v := by
  have inside := queue.valid_entry root (by simp)
  let selected : Cell matrix := FiniteGridIndex.fromIndex root.2 inside
  have readIndex : index selected = root.2 := FiniteGridIndex.index_fromIndex _ _
  have chosen := queue.minimum_current minimum waiting
  have value : distance state selected = root.1 := by
    unfold distance
    rw [readIndex]
    exact chosen.1.symm
  refine ⟨selected, readIndex, value, ?_, ?_, ?_⟩
  · simpa only [visited, readIndex] using waiting
  · rw [value]
    exact queue.finite root (by simp)
  · intro v pending
    rw [value]
    exact chosen.2 (index v) (FiniteGridIndex.index_lt v) pending

theorem queue_selected_minimum (matrix : List (List Int))
    (nonnegative : ∀ row ∈ matrix, ∀ value ∈ row, 0 ≤ value)
    (source : Cell matrix) (state : SearchState) (invariant : Invariant matrix source state)
    (root : Int × Nat) (rest : List (Int × Nat)) (queue : QueueInvariant matrix state (root :: rest))
    (minimum : ∀ entry ∈ root :: rest, root.1 ≤ entry.1) (waiting : state.visited[root.2]! = false) :
    ∃ selected : Cell matrix, index selected = root.2 ∧
      ∃ shortest : (grid matrix).Walk selected source,
        (shortest.support.map (cost matrix)).sum = root.1 ∧
        ∀ walk : (grid matrix).Walk selected source,
          (shortest.support.map (cost matrix)).sum ≤ (walk.support.map (cost matrix)).sum := by
  obtain ⟨selected, readIndex, value, pending, finite, least⟩ := queue_selected matrix state root rest queue minimum waiting
  obtain ⟨shortest, weight, optimal⟩ := DijkstraFrontier.selected_minimum (grid matrix) (cost matrix)
    (distance state) (visited state) (GridShortestPaths.cost_nonnegative matrix nonnegative)
    source selected invariant.source_bound invariant.settled invariant.relaxed pending least
    (bigVal matrix) finite (invariant.witness selected)
  exact ⟨selected, readIndex, shortest, weight.trans value, optimal⟩

/-- Python omits the visited-neighbor guard: a witnessed candidate cannot improve a settled cell. -/
theorem settled_neighbor_le_candidate (matrix : List (List Int)) (source : Cell matrix)
    (state : SearchState) (invariant : Invariant matrix source state) (selected neighbor : Cell matrix)
    (finite : distance state selected < bigVal matrix) (edge : (grid matrix).Adj neighbor selected)
    (done : visited state neighbor = true) :
    distance state neighbor ≤ distance state selected + cost matrix neighbor := by
  rcases (invariant.witness selected).2 with sentinel | ⟨walk, value⟩
  · omega
  · have bound := invariant.settled neighbor done (.cons edge walk)
    simpa only [SimpleGraph.Walk.support_cons, List.map_cons, List.sum_cons, value, add_comm] using bound

end ProjectEulerSolutions.DijkstraGrid
