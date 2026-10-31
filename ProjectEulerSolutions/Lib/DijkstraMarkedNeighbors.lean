import ProjectEulerSolutions.Lib.DijkstraQueueSelection

/-! Marking the selected vertex does not invalidate the settled-neighbor bound. -/
namespace ProjectEulerSolutions.DijkstraGrid
open ProjectEulerSolutions.P83
open GridShortestPaths (Cell cost)
open FiniteGridIndex (index)

theorem marked_neighbor_le_candidate (matrix : List (List Int)) (source : Cell matrix)
    (state : SearchState) (invariant : Invariant matrix source state) (selected neighbor : Cell matrix)
    (finite : distance state selected < ProjectEulerStatements.P83.bigVal matrix)
    (edge : (grid matrix).Adj neighbor selected)
    (done : (mark state selected)[index neighbor]! = true) :
    distance state neighbor ≤ distance state selected + cost matrix neighbor := by
  have different : neighbor ≠ selected := by
    intro same
    subst neighbor
    exact (grid matrix).loopless.irrefl selected edge
  rw [mark_read matrix state invariant.visited_size selected neighbor, if_neg different] at done
  exact settled_neighbor_le_candidate matrix source state invariant selected neighbor finite edge done

end ProjectEulerSolutions.DijkstraGrid
