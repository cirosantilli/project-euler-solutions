import ProjectEulerSolutions.Lib.DijkstraGridState

/-! Every finite array estimate continues to represent an actual grid walk. -/
namespace ProjectEulerSolutions.DijkstraGrid
open ProjectEulerSolutions.P83
open ProjectEulerStatements.P83 (bigVal)
open FiniteGridGraph (encode adjacent)
open FiniteGridIndex (index)
open GridShortestPaths (Cell cost)

private theorem min_witness {α : Type} (graph : SimpleGraph α) (cost : α → Int)
    (infinity : Int) (source v : α) (old : Int)
    (known : VertexRelaxation.Witness graph cost infinity source v old)
    (walk : graph.Walk v source) :
    VertexRelaxation.Witness graph cost infinity source v (min old (walk.support.map cost).sum) := by
  by_cases smaller : old ≤ (walk.support.map cost).sum
  · rw [min_eq_left smaller]
    exact known
  · rw [min_eq_right (by omega)]
    exact ⟨by have bound := known.1; omega, Or.inr ⟨walk, rfl⟩⟩

theorem relax_witness (matrix : List (List Int)) (source selected u : Cell matrix)
    (flags : Array Bool) (value : Int) (distances : Array Int)
    (size : distances.size = total matrix)
    (known : ∀ v : Cell matrix, VertexRelaxation.Witness (grid matrix) (cost matrix) (bigVal matrix)
      source v distances[index v]!)
    (walk : (grid matrix).Walk selected source) (walkValue : (walk.support.map (cost matrix)).sum = value)
    (edge : (grid matrix).Adj u selected) :
    ∀ v : Cell matrix, VertexRelaxation.Witness (grid matrix) (cost matrix) (bigVal matrix)
      source v (relax matrix (matrix.headD []).length flags value distances (encode u))[index v]! := by
  intro v
  rw [DijkstraArrayRelaxation.relax_read matrix _ flags value distances _ (index v)
    (by rw [size]; exact FiniteGridIndex.index_lt v), FiniteGridIndex.index_of_encode]
  split
  · rename_i affected
    have same : u = v := FiniteGridIndex.index_injective _ _ affected.1
    subst u
    have extended := min_witness (grid matrix) (cost matrix) (bigVal matrix) source v
      distances[index v]! (known v) (.cons edge walk)
    simpa only [SimpleGraph.Walk.support_cons, List.map_cons, List.sum_cons, walkValue,
      cost, GridRelaxationAgreement.cost, add_comm] using extended
  · exact known v

theorem fold_witness (matrix : List (List Int)) (source selected : Cell matrix)
    (flags : Array Bool) (value : Int) (walk : (grid matrix).Walk selected source)
    (walkValue : (walk.support.map (cost matrix)).sum = value)
    (cells : List (Cell matrix)) (edges : ∀ u ∈ cells, (grid matrix).Adj u selected)
    (distances : Array Int) (size : distances.size = total matrix)
    (known : ∀ v : Cell matrix, VertexRelaxation.Witness (grid matrix) (cost matrix) (bigVal matrix)
      source v distances[index v]!) :
    ∀ v : Cell matrix, VertexRelaxation.Witness (grid matrix) (cost matrix) (bigVal matrix)
      source v ((cells.map encode).foldl
        (relax matrix (matrix.headD []).length flags value) distances)[index v]! := by
  induction cells generalizing distances with
  | nil => exact known
  | cons u rest ih =>
      simp only [List.map_cons, List.foldl_cons]
      exact ih (fun v member => edges v (by simp [member])) _
        (by rw [DijkstraArrayRelaxation.relax_size, size])
        (relax_witness matrix source selected u flags value distances size known walk walkValue
          (edges u (by simp)))

theorem update_witness (matrix : List (List Int)) (source : Cell matrix) (state : SearchState)
    (invariant : Invariant matrix source state) (selected : Cell matrix)
    (finite : distance state selected < bigVal matrix) :
    ∀ v, VertexRelaxation.Witness (grid matrix) (cost matrix) (bigVal matrix) source v
      (distance (update matrix state selected) v) := by
  rcases (invariant.witness selected).2 with sentinel | ⟨walk, value⟩
  · omega
  · have result := fold_witness matrix source selected (mark state selected) (distance state selected)
      walk value (adjacent selected)
      (fun u member => (grid matrix).symm ((FiniteGridGraph.adjacent_iff selected u).mp member))
      state.distances invariant.distances_size invariant.witness
    rw [FiniteGridGraph.adjacent_encode] at result
    exact result

end ProjectEulerSolutions.DijkstraGrid
