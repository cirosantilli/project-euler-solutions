import ProjectEulerSolutions.Lib.GridPathBounds
import Mathlib.Combinatorics.SimpleGraph.Paths

/-! Erasing cycles justifies input-derived bounds without evaluating a shortest-path search. -/
namespace ProjectEulerSolutions.NonnegativeWalkBounds

/-- A distinct-vertex subcollection cannot cost more than the original nonnegative collection. -/
theorem cost_le_of_nodup_subset {α : Type} (cost : α → Int) (selected original : List α)
    (simple : selected.Nodup) (included : selected ⊆ original)
    (nonnegative : ∀ vertex ∈ original, 0 ≤ cost vertex) :
    (selected.map cost).sum ≤ (original.map cost).sum := by
  obtain ⟨ordered, permutation, sublist⟩ := List.subperm_iff.mp (simple.subperm included)
  have lower := (sublist.map cost).sum_le_sum (by
    intro value member
    obtain ⟨vertex, present, rfl⟩ := List.mem_map.mp member
    exact nonnegative vertex (permutation.mem_iff.mp present))
  simpa only [(permutation.map cost).sum_eq] using lower

/-- Every walk has a path with the same endpoints and no larger vertex cost, including endpoints. -/
theorem erase_cycles {α : Type} [DecidableEq α] (graph : SimpleGraph α)
    (cost : α → Int) {start finish : α} (walk : graph.Walk start finish)
    (nonnegative : ∀ vertex ∈ walk.support, 0 ≤ cost vertex) :
    ∃ path : graph.Walk start finish, path.IsPath ∧
      (path.support.map cost).sum ≤ (walk.support.map cost).sum := by
  refine ⟨walk.bypass, walk.bypass_isPath, ?_⟩
  exact cost_le_of_nodup_subset cost _ _ walk.bypass_isPath.support_nodup
    walk.support_bypass_subset nonnegative

/-- All finite nonnegative vertex-weighted graphs admit the same sum and vertex-count bounds. -/
theorem bounded_path_enumerated {α : Type} [DecidableEq α] (vertices : List α)
    (complete : ∀ vertex, vertex ∈ vertices) (graph : SimpleGraph α) (cost : α → Int)
    (nonnegative : ∀ vertex, 0 ≤ cost vertex)
    {start finish : α} (walk : graph.Walk start finish) :
    ∃ path : graph.Walk start finish, path.IsPath ∧
      (path.support.map cost).sum ≤ (walk.support.map cost).sum ∧
      path.length < vertices.length ∧
      (path.support.map cost).sum < (vertices.map cost).sum + 1 := by
  obtain ⟨path, simple, cheaper⟩ := erase_cycles graph cost walk (fun vertex _ => nonnegative vertex)
  have included : path.support ⊆ vertices := fun vertex _ => complete vertex
  have lengthBound := (simple.support_nodup.subperm included).length_le
  have costBound := cost_le_of_nodup_subset cost path.support vertices simple.support_nodup
    included (fun vertex _ => nonnegative vertex)
  refine ⟨path, simple, cheaper, ?_, by omega⟩
  simpa using lengthBound

/-- The flat-index specialization used by array implementations. -/
theorem bounded_path (weights : List Int) (graph : SimpleGraph (Fin weights.length))
    (nonnegative : ∀ value ∈ weights, 0 ≤ value)
    {start finish : Fin weights.length} (walk : graph.Walk start finish) :
    ∃ path : graph.Walk start finish, path.IsPath ∧
      (path.support.map weights.get).sum ≤ (walk.support.map weights.get).sum ∧
      path.length < weights.length ∧
      (path.support.map weights.get).sum < weights.sum + 1 := by
  obtain ⟨path, simple, cheaper⟩ := erase_cycles graph weights.get walk
    (fun vertex _ => nonnegative _ (List.get_mem weights vertex))
  have bound := GridPathBounds.simple_path_bound weights path.support nonnegative simple.support_nodup
  exact ⟨path, simple, cheaper, by simpa using bound.1, bound.2⟩

end ProjectEulerSolutions.NonnegativeWalkBounds
