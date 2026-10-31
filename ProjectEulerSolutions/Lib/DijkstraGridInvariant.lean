import ProjectEulerSolutions.Lib.DijkstraGridWitness

/-! Preservation of the shortest-path invariant by the array Dijkstra step. -/
namespace ProjectEulerSolutions.DijkstraGrid
open ProjectEulerSolutions.P83
open ProjectEulerStatements.P83 (bigVal)
open FiniteGridGraph (encode adjacent)
open FiniteGridIndex (index)
open GridShortestPaths (Cell cost)

theorem update_invariant (matrix : List (List Int))
    (nonnegative : ∀ row ∈ matrix, ∀ value ∈ row, 0 ≤ value)
    (source : Cell matrix) (state : SearchState) (invariant : Invariant matrix source state)
    (selected : Cell matrix) (pending : visited state selected = false)
    (finite : distance state selected < bigVal matrix)
    (least : ∀ u : Cell matrix, visited state u = false → distance state selected ≤ distance state u) :
    Invariant matrix source (update matrix state selected) := by
  have optimal := DijkstraFrontier.selected_le_walk (grid matrix) (cost matrix)
    (distance state) (visited state) (GridShortestPaths.cost_nonnegative matrix nonnegative)
    source selected invariant.source_bound invariant.settled invariant.relaxed pending least
  have marked (v : Cell matrix) : visited (update matrix state selected) v =
      if v = selected then true else visited state v :=
    mark_read matrix state invariant.visited_size selected v
  have fixed (v : Cell matrix) (done : visited (update matrix state selected) v = true) :
      distance (update matrix state selected) v = distance state v :=
    update_settled matrix state invariant.distances_size selected v done
  refine ⟨(update_sizes matrix state selected).1.trans invariant.visited_size,
    (update_sizes matrix state selected).2.trans invariant.distances_size,
    (update_le matrix state invariant.distances_size selected source).trans invariant.source_bound,
    update_witness matrix source state invariant selected finite, ?_, ?_⟩
  · intro v done walk
    rw [fixed v done]
    by_cases same : v = selected
    · subst v
      exact optimal walk
    · exact invariant.settled v (by simpa only [marked, if_neg same] using done) walk
  · intro v u edge waiting done
    have unequal : v ≠ selected := by intro same; simp [marked, same] at waiting
    have oldWaiting : visited state v = false := by simpa only [marked, if_neg unequal] using waiting
    rw [fixed u done]
    by_cases same : u = selected
    · subst u
      exact update_neighbor matrix state invariant.distances_size selected v edge waiting
    · have oldDone : visited state u = true := by simpa only [marked, if_neg same] using done
      exact (update_le matrix state invariant.distances_size selected v).trans
        (invariant.relaxed v u edge oldWaiting oldDone)

theorem searchStep_invariant (matrix : List (List Int))
    (nonnegative : ∀ row ∈ matrix, ∀ value ∈ row, 0 ≤ value)
    (source : Cell matrix) (state : SearchState) (invariant : Invariant matrix source state) :
    Invariant matrix source (searchStep matrix state) := by
  let chosen := findMin (total matrix) (bigVal matrix) state
  by_cases sentinel : chosen.2 = bigVal matrix
  · have same : searchStep matrix state = state := by
      dsimp only [searchStep]
      rw [if_pos sentinel]
    rw [same]
    exact invariant
  · have finite : chosen.2 < bigVal matrix := lt_of_le_of_ne
      (DijkstraSelection.findMin_le (total matrix) (bigVal matrix) state) sentinel
    have selected := DijkstraSelection.findMin_finite (total matrix) (bigVal matrix) state finite
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
    rw [searchStep_eq_update matrix state v chosenPair (by rw [value]; exact finite)]
    apply update_invariant matrix nonnegative source state invariant v pending (by rw [value]; exact finite)
    intro u waiting
    rw [value]
    exact DijkstraSelection.findMin_le_unvisited (total matrix) (bigVal matrix) state (index u)
      (FiniteGridIndex.index_lt u) waiting

end ProjectEulerSolutions.DijkstraGrid
