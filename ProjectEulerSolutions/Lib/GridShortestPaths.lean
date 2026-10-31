import ProjectEulerSolutions.Lib.FiniteGridGraph
import ProjectEulerSolutions.Lib.GridRelaxationAgreement
import Mathlib.Data.List.GetD

/-! The four-direction matrix reference attains a shortest vertex-weighted grid walk. -/
namespace ProjectEulerSolutions.GridShortestPaths
open ProjectEulerStatements.P83
open FiniteGridGraph

abbrev Cell (matrix : List (List Int)) := Vertex matrix.length (matrix.headD []).length

def cost (matrix : List (List Int)) (v : Cell matrix) : Int :=
  GridRelaxationAgreement.cost matrix (encode v)

private theorem row_values (row : List Int) (width : Nat) (shape : row.length = width) :
    (List.finRange width).map (fun c => row.getD c.val 0) = row := by
  subst width
  calc
    _ = (List.finRange row.length).map row.get := by
      apply List.map_congr_left
      intro c _
      exact List.getD_eq_get row 0 c
    _ = row := List.map_get_finRange row

/-- Enumerating grid costs row by row gives exactly the original matrix entries. -/
theorem vertices_cost (matrix : List (List Int))
    (shape : ∀ row ∈ matrix, row.length = (matrix.headD []).length) :
    (vertices matrix.length (matrix.headD []).length).map (cost matrix) = matrix.flatten := by
  calc
    _ = (List.finRange matrix.length).flatMap matrix.get := by
      simp only [vertices, List.map_flatMap, List.map_map, Function.comp_def]
      simp only [List.flatMap_def]
      apply congrArg List.flatten
      apply List.map_congr_left
      intro r _
      simp only [cost, GridRelaxationAgreement.cost, encode, getD2,
        Int.natCast_nonneg, and_self, if_true, Int.toNat_natCast,
        List.getD_eq_get matrix [] r]
      exact row_values (matrix.get r) _ (shape _ (List.get_mem matrix r))
    _ = matrix.flatten := by rw [List.flatMap_def, List.map_get_finRange]

theorem cost_nonnegative (matrix : List (List Int))
    (nonnegative : ∀ row ∈ matrix, ∀ value ∈ row, 0 ≤ value) (v : Cell matrix) :
    0 ≤ cost matrix v :=
  GridRelaxationAgreement.getD2_nonnegative matrix nonnegative (encode v).1 (encode v).2

theorem run_agreement (matrix : List (List Int)) (source : Cell matrix) (n : Nat) (v : Cell matrix) :
    VertexRelaxation.run (cost matrix) adjacent (bigVal matrix) source n v =
      VertexRelaxation.run (GridRelaxationAgreement.cost matrix)
        (GridRelaxationAgreement.adjacent matrix) (bigVal matrix) (encode source) n (encode v) := by
  exact VertexRelaxation.run_map encode (encode_injective _ _) (cost matrix)
    (GridRelaxationAgreement.cost matrix) (fun _ => rfl) adjacent
    (GridRelaxationAgreement.adjacent matrix) adjacent_encode _ source n v

/-- Every cell's reference distance is the attained minimum among all walks to the origin. -/
theorem rounds_minimum (matrix : List (List Int))
    (shape : ∀ row ∈ matrix, row.length = (matrix.headD []).length)
    (nonnegative : ∀ row ∈ matrix, ∀ value ∈ row, 0 ≤ value)
    (rowsPositive : 0 < matrix.length) (colsPositive : 0 < (matrix.headD []).length)
    (v : Cell matrix) :
    ∃ shortest : (graph matrix.length (matrix.headD []).length).Walk v (origin rowsPositive colsPositive),
      (shortest.support.map (cost matrix)).sum =
        GridRelaxationAgreement.cost
          (GridRelaxationAgreement.rounds matrix (matrix.length * (matrix.headD []).length)) (encode v) ∧
      ∀ walk : (graph matrix.length (matrix.headD []).length).Walk v (origin rowsPositive colsPositive),
        (shortest.support.map (cost matrix)).sum ≤ (walk.support.map (cost matrix)).sum := by
  obtain ⟨walk⟩ := reachable_origin rowsPositive colsPositive v
  have result := VertexRelaxation.run_minimum_enumerated
    (vertices matrix.length (matrix.headD []).length) mem_vertices
    (graph matrix.length (matrix.headD []).length) (cost matrix) adjacent adjacent_iff
    (cost_nonnegative matrix nonnegative) (origin rowsPositive colsPositive) v walk
  rw [vertices_cost matrix shape, vertices_length, ← GridPathBounds.bigVal_eq_sum] at result
  rw [run_agreement] at result
  have inside : GridRelaxationAgreement.valid matrix (encode v) := by
    unfold GridRelaxationAgreement.valid encode
    have := v.1.isLt
    have := v.2.isLt
    omega
  rw [GridRelaxationAgreement.rounds_eq_run matrix nonnegative _ _ inside]
  exact result

def finish (matrix : List (List Int)) (rowsPositive : 0 < matrix.length)
    (colsPositive : 0 < (matrix.headD []).length) : Cell matrix :=
  (⟨matrix.length - 1, by omega⟩, ⟨(matrix.headD []).length - 1, by omega⟩)

/-- The reference answer is the cost of a minimum walk between the opposite corners. -/
theorem naive_minimum (matrix : List (List Int))
    (shape : ∀ row ∈ matrix, row.length = (matrix.headD []).length)
    (nonnegative : ∀ row ∈ matrix, ∀ value ∈ row, 0 ≤ value)
    (rowsPositive : 0 < matrix.length) (colsPositive : 0 < (matrix.headD []).length) :
    ∃ shortest : (graph matrix.length (matrix.headD []).length).Walk
        (finish matrix rowsPositive colsPositive) (origin rowsPositive colsPositive),
      (shortest.support.map (cost matrix)).sum = naive matrix ∧
      ∀ walk : (graph matrix.length (matrix.headD []).length).Walk
          (finish matrix rowsPositive colsPositive) (origin rowsPositive colsPositive),
        (shortest.support.map (cost matrix)).sum ≤ (walk.support.map (cost matrix)).sum := by
  have result := rounds_minimum matrix shape nonnegative rowsPositive colsPositive
    (finish matrix rowsPositive colsPositive)
  have coordinates : encode (finish matrix rowsPositive colsPositive) =
      ((matrix.length : Int) - 1, ((matrix.headD []).length : Int) - 1) := by
    ext <;> simp only [encode, finish] <;> omega
  simpa only [naive, minPath, dims, GridRelaxationAgreement.iterate_eq_rounds,
    GridRelaxationAgreement.cost, coordinates] using result

end ProjectEulerSolutions.GridShortestPaths
