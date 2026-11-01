import ProjectEulerStatements.P83
import Mathlib.Algebra.Order.BigOperators.Group.List
import Mathlib.Data.List.FinRange
import Mathlib.Data.List.Perm.Subperm

/-! Input-derived distance and iteration bounds for finite nonnegative grids. -/
namespace ProjectEulerSolutions.GridPathBounds

/-- A path visiting distinct vertices uses at most all vertices and costs at most their sum.
The order in which the path visits them is unrestricted, and distinct vertices may have equal costs. -/
theorem simple_path_bound (weights : List Int) (path : List (Fin weights.length))
    (nonnegative : ∀ value ∈ weights, 0 ≤ value) (simple : path.Nodup) :
    path.length ≤ weights.length ∧ (path.map weights.get).sum < weights.sum + 1 := by
  have included : path.Subperm (List.finRange weights.length) :=
    simple.subperm (by intro index _; simp)
  obtain ⟨ordered, permutation, sublist⟩ := List.subperm_iff.mp included
  have costLe : (path.map weights.get).sum ≤ (ordered.map weights.get).sum :=
    (sublist.map weights.get).sum_le_sum (by
      intro value member
      obtain ⟨index, _, rfl⟩ := List.mem_map.mp member
      exact nonnegative _ (List.get_mem weights index))
  have total : (ordered.map weights.get).sum = weights.sum := by
    rw [(permutation.map weights.get).sum_eq, List.map_get_finRange]
  refine ⟨by simpa using included.length_le, ?_⟩
  rw [total] at costLe
  omega

theorem bigVal_eq_sum (matrix : List (List Int)) :
    ProjectEulerStatements.P83.bigVal matrix = matrix.flatten.sum + 1 := by
  rw [List.sum_flatten]
  simp only [ProjectEulerStatements.P83.bigVal, List.sum_eq_foldl, List.foldl_map]

theorem rectangular_size (matrix : List (List Int)) (width : Nat)
    (shape : ∀ row ∈ matrix, row.length = width) :
    matrix.flatten.length = matrix.length * width := by
  induction matrix with
  | nil => simp
  | cons row rows ih =>
      have head := shape row (by simp)
      have tail := ih (fun other member => shape other (by simp [member]))
      simpa [List.flatten_cons, head, Nat.add_mul, Nat.add_comm] using
        congrArg (fun n => row.length + n) tail

theorem matrix_path_bound (matrix : List (List Int))
    (nonnegative : ∀ row ∈ matrix, ∀ value ∈ row, 0 ≤ value)
    (shape : ∀ row ∈ matrix, row.length = (matrix.headD []).length)
    (path : List (Fin matrix.flatten.length)) (simple : path.Nodup) :
    path.length ≤ matrix.length * (matrix.headD []).length ∧
      (path.map matrix.flatten.get).sum < ProjectEulerStatements.P83.bigVal matrix := by
  have positive : ∀ value ∈ matrix.flatten, 0 ≤ value := by
    intro value member
    obtain ⟨row, inMatrix, inRow⟩ := List.mem_flatten.mp member
    exact nonnegative row inMatrix value inRow
  have bound := simple_path_bound matrix.flatten path positive simple
  rw [bigVal_eq_sum]
  exact ⟨by simpa only [rectangular_size matrix _ shape] using bound.1, bound.2⟩

end ProjectEulerSolutions.GridPathBounds
