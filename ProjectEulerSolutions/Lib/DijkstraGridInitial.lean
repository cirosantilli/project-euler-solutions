import ProjectEulerSolutions.Lib.DijkstraGridState

/-! The initial P83 arrays satisfy the grid shortest-path invariant. -/
namespace ProjectEulerSolutions.DijkstraGrid
open ProjectEulerSolutions.P83
open ProjectEulerStatements.P83 (bigVal getD2)
open FiniteGridGraph (encode adjacent origin vertices)
open FiniteGridIndex (index)
open GridShortestPaths (Cell cost)

def initial (matrix : List (List Int)) : SearchState :=
  ⟨Array.replicate (total matrix) false,
    (Array.replicate (total matrix) (bigVal matrix)).set! 0 (getD2 matrix 0 0)⟩

theorem cost_lt_bound (matrix : List (List Int))
    (shape : ∀ row ∈ matrix, row.length = (matrix.headD []).length)
    (nonnegative : ∀ row ∈ matrix, ∀ value ∈ row, 0 ≤ value) (v : Cell matrix) :
    cost matrix v < bigVal matrix := by
  have bound := NonnegativeWalkBounds.cost_le_of_nodup_subset (cost matrix) [v]
    (vertices matrix.length (matrix.headD []).length) (by simp)
    (by simpa using FiniteGridGraph.mem_vertices v)
    (fun u _ => GridShortestPaths.cost_nonnegative matrix nonnegative u)
  rw [GridShortestPaths.vertices_cost matrix shape] at bound
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero] at bound
  rw [GridPathBounds.bigVal_eq_sum]
  omega

theorem initial_visited (matrix : List (List Int)) (v : Cell matrix) :
    visited (initial matrix) v = false := by
  simp [visited, initial, getElem!_pos, FiniteGridIndex.index_lt v]

theorem initial_distance (matrix : List (List Int))
    (rowsPositive : 0 < matrix.length) (colsPositive : 0 < (matrix.headD []).length) (v : Cell matrix) :
    distance (initial matrix) v =
      VertexRelaxation.initial (cost matrix) (bigVal matrix) (origin rowsPositive colsPositive) v := by
  have inside := FiniteGridIndex.index_lt v
  have zero : index (origin rowsPositive colsPositive) = 0 := by simp [index, origin]
  have same : 0 = index v ↔ v = origin rowsPositive colsPositive := by
    rw [← zero]
    simp only [(FiniteGridIndex.index_injective _ _).eq_iff, eq_comm]
  simp only [distance, initial, Array.set!_eq_setIfInBounds, getElem!_pos,
    Array.size_setIfInBounds, Array.size_replicate, inside, Array.getElem_setIfInBounds,
    Array.getElem_replicate, same, VertexRelaxation.initial]
  rfl

theorem initial_invariant (matrix : List (List Int))
    (shape : ∀ row ∈ matrix, row.length = (matrix.headD []).length)
    (nonnegative : ∀ row ∈ matrix, ∀ value ∈ row, 0 ≤ value)
    (rowsPositive : 0 < matrix.length) (colsPositive : 0 < (matrix.headD []).length) :
    Invariant matrix (origin rowsPositive colsPositive) (initial matrix) := by
  refine ⟨by simp [initial], by simp [initial], ?_, ?_, ?_, ?_⟩
  · rw [initial_distance matrix rowsPositive colsPositive]
    simp [VertexRelaxation.initial]
  · intro v
    rw [initial_distance matrix rowsPositive colsPositive]
    exact VertexRelaxation.run_witness (grid matrix) (cost matrix) adjacent
      (fun u v member => (FiniteGridGraph.adjacent_iff u v).mp member)
      (GridShortestPaths.cost_nonnegative matrix nonnegative) (bigVal matrix)
      (origin rowsPositive colsPositive)
      (le_of_lt (cost_lt_bound matrix shape nonnegative _)) 0 v
  · intro v done
    rw [initial_visited] at done
    cases done
  · intro v u _ _ done
    rw [initial_visited] at done
    cases done

end ProjectEulerSolutions.DijkstraGrid
