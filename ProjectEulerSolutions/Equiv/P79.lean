import ProjectEulerSolutions.P79
namespace ProjectEulerSolutions.P79
open ProjectEulerStatements.P79 ProjectEulerStatements.TopologicalOrder

theorem equiv (attempts : List String) : ProjectEulerStatements.P79.naive attempts = solve attempts := by
  unfold ProjectEulerStatements.P79.naive solve
  cases checked : observations attempts with
  | error error => rfl
  | ok triples =>
    change output (order (vertices triples) (expandedEdges triples)) =
      output (order (vertices triples) (adjacentEdges triples))
    congr 1
    apply order_expansion
    intro triple member
    have first : triple.1 ∈ vertices triples := by
      simp only [vertices, List.mem_eraseDups, List.mem_flatMap]
      exact ⟨triple, member, by simp⟩
    have second : triple.2.1 ∈ vertices triples := by
      simp only [vertices, List.mem_eraseDups, List.mem_flatMap]
      exact ⟨triple, member, by simp⟩
    have third : triple.2.2 ∈ vertices triples := by
      simp only [vertices, List.mem_eraseDups, List.mem_flatMap]
      exact ⟨triple, member, by simp⟩
    exact ⟨first, second, third⟩
end ProjectEulerSolutions.P79
