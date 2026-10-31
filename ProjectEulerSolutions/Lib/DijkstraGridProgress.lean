import ProjectEulerSolutions.Lib.DijkstraGridInvariant
import Mathlib.Data.Fintype.Prod
import Mathlib.Data.Finset.Card

/-! Each active Dijkstra step settles one new cell; the sentinel cannot stop a reachable grid early. -/
namespace ProjectEulerSolutions.DijkstraGrid
open ProjectEulerSolutions.P83
open ProjectEulerStatements.P83 (bigVal)
open FiniteGridGraph (encode adjacent vertices)
open FiniteGridIndex (index)
open GridShortestPaths (Cell cost)

theorem finite_available (matrix : List (List Int))
    (shape : ∀ row ∈ matrix, row.length = (matrix.headD []).length)
    (nonnegative : ∀ row ∈ matrix, ∀ value ∈ row, 0 ≤ value)
    (source : Cell matrix) (reachable : ∀ v : Cell matrix, Nonempty ((grid matrix).Walk v source))
    (state : SearchState) (invariant : Invariant matrix source state)
    (pending : ∃ v : Cell matrix, visited state v = false) :
    ∃ i < total matrix, state.visited[i]! = false ∧ state.distances[i]! < bigVal matrix := by
  obtain ⟨v, waiting⟩ := pending
  obtain ⟨walk⟩ := reachable v
  obtain ⟨path, _, _, _, below⟩ := NonnegativeWalkBounds.bounded_path_enumerated
    (vertices matrix.length (matrix.headD []).length) FiniteGridGraph.mem_vertices
    (grid matrix) (cost matrix) (GridShortestPaths.cost_nonnegative matrix nonnegative) walk
  rw [GridShortestPaths.vertices_cost matrix shape, ← GridPathBounds.bigVal_eq_sum] at below
  obtain ⟨u, unvisited, lower⟩ := DijkstraFrontier.frontier_bound (grid matrix) (cost matrix)
    (distance state) (visited state) (GridShortestPaths.cost_nonnegative matrix nonnegative)
    source invariant.source_bound invariant.settled invariant.relaxed path waiting
  exact ⟨index u, FiniteGridIndex.index_lt u, unvisited, lower.trans_lt below⟩

def pendingCells (matrix : List (List Int)) (state : SearchState) : Finset (Cell matrix) :=
  Finset.univ.filter (fun v => visited state v = false)

theorem mem_pendingCells (matrix : List (List Int)) (state : SearchState) (v : Cell matrix) :
    v ∈ pendingCells matrix state ↔ visited state v = false := by simp [pendingCells]

theorem pending_update (matrix : List (List Int)) (state : SearchState)
    (size : state.visited.size = total matrix) (selected : Cell matrix) :
    pendingCells matrix (update matrix state selected) = (pendingCells matrix state).erase selected := by
  ext v
  simp only [mem_pendingCells, Finset.mem_erase]
  change (mark state selected)[index v]! = false ↔ _
  rw [mark_read matrix state size selected v]
  by_cases same : v = selected <;> simp [same]

theorem pending_le_total (matrix : List (List Int)) (state : SearchState) :
    (pendingCells matrix state).card ≤ total matrix := by
  have bound := Finset.card_filter_le (Finset.univ : Finset (Cell matrix)) (fun v => visited state v = false)
  simpa only [pendingCells, Finset.card_univ, Fintype.card_prod, Fintype.card_fin] using bound

theorem pending_progress (matrix : List (List Int))
    (shape : ∀ row ∈ matrix, row.length = (matrix.headD []).length)
    (nonnegative : ∀ row ∈ matrix, ∀ value ∈ row, 0 ≤ value)
    (source : Cell matrix) (reachable : ∀ v : Cell matrix, Nonempty ((grid matrix).Walk v source))
    (state : SearchState) (invariant : Invariant matrix source state)
    (positive : 0 < (pendingCells matrix state).card) :
    (pendingCells matrix (searchStep matrix state)).card + 1 = (pendingCells matrix state).card := by
  obtain ⟨u, member⟩ := Finset.card_pos.mp positive
  have available := finite_available matrix shape nonnegative source reachable state invariant
    ⟨u, (mem_pendingCells matrix state u).mp member⟩
  let chosen := findMin (total matrix) (bigVal matrix) state
  have selected := DijkstraSelection.findMin_selected (total matrix) (bigVal matrix) state available
  let v : Cell matrix := FiniteGridIndex.fromIndex chosen.1 selected.1
  have readIndex : index v = chosen.1 := FiniteGridIndex.index_fromIndex _ _
  have value : distance state v = chosen.2 := by
    unfold distance
    rw [readIndex]
    exact selected.2.2.1.symm
  have pending : visited state v = false := by
    unfold visited
    rw [readIndex]
    exact selected.2.1
  have chosenPair : findMin (total matrix) (bigVal matrix) state = (index v, distance state v) := by
    rw [readIndex, value]
  rw [searchStep_eq_update matrix state v chosenPair (by rw [value]; exact selected.2.2.2),
    pending_update matrix state invariant.visited_size v]
  exact Finset.card_erase_add_one ((mem_pendingCells matrix state v).mpr pending)

theorem pending_step_le (matrix : List (List Int)) (source : Cell matrix) (state : SearchState)
    (invariant : Invariant matrix source state) :
    (pendingCells matrix (searchStep matrix state)).card ≤ (pendingCells matrix state).card := by
  let chosen := findMin (total matrix) (bigVal matrix) state
  by_cases sentinel : chosen.2 = bigVal matrix
  · have same : searchStep matrix state = state := by
      dsimp only [searchStep]
      rw [if_pos sentinel]
    rw [same]
  · have finite : chosen.2 < bigVal matrix := lt_of_le_of_ne
      (DijkstraSelection.findMin_le (total matrix) (bigVal matrix) state) sentinel
    have selected := DijkstraSelection.findMin_finite (total matrix) (bigVal matrix) state finite
    let v : Cell matrix := FiniteGridIndex.fromIndex chosen.1 selected.1
    have readIndex : index v = chosen.1 := FiniteGridIndex.index_fromIndex _ _
    have value : distance state v = chosen.2 := by
      unfold distance
      rw [readIndex]
      exact selected.2.2.1.symm
    have chosenPair : findMin (total matrix) (bigVal matrix) state = (index v, distance state v) := by
      rw [readIndex, value]
    rw [searchStep_eq_update matrix state v chosenPair (by rw [value]; exact finite),
      pending_update matrix state invariant.visited_size v]
    exact Finset.card_le_card (Finset.erase_subset _ _)

end ProjectEulerSolutions.DijkstraGrid
