import ProjectEulerSolutions.Lib.VertexRelaxation

/-! The frontier argument for Dijkstra with nonnegative vertex costs.
Every walk from an unvisited vertex encounters either the source or an edge
from an unvisited vertex to a settled one. -/
namespace ProjectEulerSolutions.DijkstraFrontier

variable {α : Type}

/-- A walk contains an unvisited frontier estimate no greater than its full cost. -/
theorem frontier_bound (graph : SimpleGraph α) (cost distance : α → Int)
    (visited : α → Bool) (nonnegative : ∀ v, 0 ≤ cost v) (source : α)
    (sourceBound : distance source ≤ cost source)
    (settled : ∀ u, visited u = true → ∀ walk : graph.Walk u source,
      distance u ≤ (walk.support.map cost).sum)
    (relaxed : ∀ v u, graph.Adj v u → visited v = false → visited u = true →
      distance v ≤ cost v + distance u)
    {v : α} (walk : graph.Walk v source) (unvisited : visited v = false) :
    ∃ u, visited u = false ∧ distance u ≤ (walk.support.map cost).sum := by
  cases walk with
  | nil => exact ⟨source, unvisited, by simpa using sourceBound⟩
  | @cons v u source edge tail =>
      cases state : visited u with
      | false =>
          obtain ⟨frontier, pending, bound⟩ :=
            frontier_bound graph cost distance visited nonnegative source sourceBound settled relaxed tail state
          refine ⟨frontier, pending, ?_⟩
          simp only [SimpleGraph.Walk.support_cons, List.map_cons, List.sum_cons]
          have positive := nonnegative v
          omega
      | true =>
          refine ⟨v, unvisited, ?_⟩
          have crossing := relaxed v u edge unvisited state
          have lower := settled u state tail
          simp only [SimpleGraph.Walk.support_cons, List.map_cons, List.sum_cons]
          omega
termination_by walk.length

/-- Selecting a least unvisited estimate makes it a lower bound for every competing walk. -/
theorem selected_le_walk (graph : SimpleGraph α) (cost distance : α → Int)
    (visited : α → Bool) (nonnegative : ∀ v, 0 ≤ cost v) (source selected : α)
    (sourceBound : distance source ≤ cost source)
    (settled : ∀ u, visited u = true → ∀ walk : graph.Walk u source,
      distance u ≤ (walk.support.map cost).sum)
    (relaxed : ∀ v u, graph.Adj v u → visited v = false → visited u = true →
      distance v ≤ cost v + distance u)
    (pending : visited selected = false)
    (least : ∀ u, visited u = false → distance selected ≤ distance u)
    (walk : graph.Walk selected source) :
    distance selected ≤ (walk.support.map cost).sum := by
  obtain ⟨frontier, unvisited, bound⟩ :=
    frontier_bound graph cost distance visited nonnegative source sourceBound settled relaxed walk pending
  exact (least frontier unvisited).trans bound

/-- A witnessed finite estimate selected by Dijkstra attains a minimum over all walks. -/
theorem selected_minimum (graph : SimpleGraph α) (cost distance : α → Int)
    (visited : α → Bool) (nonnegative : ∀ v, 0 ≤ cost v) (source selected : α)
    (sourceBound : distance source ≤ cost source)
    (settled : ∀ u, visited u = true → ∀ walk : graph.Walk u source,
      distance u ≤ (walk.support.map cost).sum)
    (relaxed : ∀ v u, graph.Adj v u → visited v = false → visited u = true →
      distance v ≤ cost v + distance u)
    (pending : visited selected = false)
    (least : ∀ u, visited u = false → distance selected ≤ distance u)
    (infinity : Int) (finite : distance selected < infinity)
    (witness : VertexRelaxation.Witness graph cost infinity source selected (distance selected)) :
    ∃ shortest : graph.Walk selected source,
      (shortest.support.map cost).sum = distance selected ∧
      ∀ walk : graph.Walk selected source,
        (shortest.support.map cost).sum ≤ (walk.support.map cost).sum := by
  rcases witness.2 with sentinel | ⟨shortest, value⟩
  · omega
  · refine ⟨shortest, value, ?_⟩
    intro walk
    rw [value]
    exact selected_le_walk graph cost distance visited nonnegative source selected
      sourceBound settled relaxed pending least walk

end ProjectEulerSolutions.DijkstraFrontier
