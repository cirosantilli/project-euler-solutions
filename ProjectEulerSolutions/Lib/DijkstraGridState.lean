import ProjectEulerSolutions.Lib.DijkstraArrayRelaxation
import ProjectEulerSolutions.Lib.DijkstraSelection
import ProjectEulerSolutions.Lib.DijkstraFrontier
import ProjectEulerSolutions.Lib.FiniteGridIndex
import ProjectEulerSolutions.Lib.GridShortestPaths

/-! The shortest-path invariant on the actual array state used by P83. -/
namespace ProjectEulerSolutions.DijkstraGrid
open ProjectEulerSolutions.P83
open ProjectEulerStatements.P83 (bigVal getD2 neighbors)
open FiniteGridGraph (encode adjacent)
open FiniteGridIndex (index)
open GridShortestPaths (Cell cost)

abbrev total (matrix : List (List Int)) := matrix.length * (matrix.headD []).length
abbrev grid (matrix : List (List Int)) := FiniteGridGraph.graph matrix.length (matrix.headD []).length
def distance {matrix : List (List Int)} (state : SearchState) (v : Cell matrix) : Int :=
  state.distances[index v]!
def visited {matrix : List (List Int)} (state : SearchState) (v : Cell matrix) : Bool :=
  state.visited[index v]!

structure Invariant (matrix : List (List Int)) (source : Cell matrix) (state : SearchState) : Prop where
  visited_size : state.visited.size = total matrix
  distances_size : state.distances.size = total matrix
  source_bound : distance state source ≤ cost matrix source
  witness : ∀ v, VertexRelaxation.Witness (grid matrix) (cost matrix) (bigVal matrix)
    source v (distance state v)
  settled : ∀ v, visited state v = true → ∀ walk : (grid matrix).Walk v source,
    distance state v ≤ (walk.support.map (cost matrix)).sum
  relaxed : ∀ v u, (grid matrix).Adj v u → visited state v = false → visited state u = true →
    distance state v ≤ cost matrix v + distance state u

def mark {matrix : List (List Int)} (state : SearchState) (selected : Cell matrix) : Array Bool :=
  state.visited.set! (index selected) true

def update (matrix : List (List Int)) (state : SearchState) (selected : Cell matrix) : SearchState :=
  ⟨mark state selected,
    (neighbors (encode selected).1 (encode selected).2 matrix.length (matrix.headD []).length).foldl
      (relax matrix (matrix.headD []).length (mark state selected) (distance state selected)) state.distances⟩

theorem mark_read (matrix : List (List Int)) (state : SearchState)
    (size : state.visited.size = total matrix) (selected v : Cell matrix) :
    (mark state selected)[index v]! = if v = selected then true else visited state v := by
  have inside : index v < state.visited.size := by rw [size]; exact FiniteGridIndex.index_lt v
  simp only [mark, Array.set!_eq_setIfInBounds, getElem!_pos,
    Array.size_setIfInBounds, inside, Array.getElem_setIfInBounds]
  simp only [(FiniteGridIndex.index_injective _ _).eq_iff, eq_comm, visited, getElem!_pos, inside]

theorem update_sizes (matrix : List (List Int)) (state : SearchState) (selected : Cell matrix) :
    (update matrix state selected).visited.size = state.visited.size ∧
      (update matrix state selected).distances.size = state.distances.size := by
  simp [update, mark, DijkstraArrayRelaxation.fold_size]

theorem update_le (matrix : List (List Int)) (state : SearchState)
    (size : state.distances.size = total matrix) (selected v : Cell matrix) :
    distance (update matrix state selected) v ≤ distance state v :=
  DijkstraArrayRelaxation.fold_le matrix _ _ _ _ _ (index v)
    (by rw [size]; exact FiniteGridIndex.index_lt v)

theorem update_settled (matrix : List (List Int)) (state : SearchState)
    (size : state.distances.size = total matrix) (selected v : Cell matrix)
    (settled : (mark state selected)[index v]! = true) :
    distance (update matrix state selected) v = distance state v :=
  DijkstraArrayRelaxation.fold_settled matrix _ _ _ _ _ (index v)
    (by rw [size]; exact FiniteGridIndex.index_lt v) settled

theorem update_neighbor (matrix : List (List Int)) (state : SearchState)
    (size : state.distances.size = total matrix) (selected v : Cell matrix)
    (edge : (grid matrix).Adj v selected) (pending : (mark state selected)[index v]! = false) :
    distance (update matrix state selected) v ≤ cost matrix v + distance state selected := by
  have member : encode v ∈ neighbors (encode selected).1 (encode selected).2
      matrix.length (matrix.headD []).length := by
    rw [← FiniteGridGraph.adjacent_encode]
    exact List.mem_map_of_mem ((FiniteGridGraph.adjacent_iff selected v).mpr (grid matrix |>.symm edge))
  have bound := DijkstraArrayRelaxation.fold_neighbor_bound matrix (matrix.headD []).length
    (mark state selected) (distance state selected) state.distances _ (encode v) member
    (index v) (FiniteGridIndex.index_of_encode v)
    (by rw [size]; exact FiniteGridIndex.index_lt v) pending
  simpa only [update, distance, cost, GridRelaxationAgreement.cost, add_comm] using bound

theorem searchStep_eq_update (matrix : List (List Int)) (state : SearchState) (selected : Cell matrix)
    (chosen : findMin (total matrix) (bigVal matrix) state = (index selected, distance state selected))
    (finite : distance state selected < bigVal matrix) :
    searchStep matrix state = update matrix state selected := by
  simp only [searchStep, chosen, ne_of_lt finite, if_false, ← Int.natCast_ediv,
    ← Int.natCast_emod, FiniteGridIndex.index_div,
    FiniteGridIndex.index_mod]
  rfl

end ProjectEulerSolutions.DijkstraGrid
