import ProjectEulerSolutions.Lib.PrimePairGraph

namespace ProjectEulerSolutions.PrimePairCertificate.Graph

/-- Resolve the source vertex before testing neighbors. The row lookup is
independent of the destination, so kernel reduction can reuse it. -/
def lookupRow : Graph → Nat → List Nat
  | .empty, _ => []
  | .node p neighbors left right, query =>
      if query < p then lookupRow left query else
        if p < query then lookupRow right query else neighbors

def lookupEdge (graph : Graph) (p q : Nat) : Bool := (lookupRow graph p).contains q

theorem edge_eq_lookup_at (graph : Graph) (p q : Nat) : graph.edge p q = graph.lookupEdge p q := by
  induction graph with
  | empty => rfl
  | node key neighbors left right ihLeft ihRight =>
    simp only [edge, lookupEdge, lookupRow]
    by_cases before : p < key
    · simpa only [if_pos before, lookupEdge] using ihLeft
    · by_cases after : key < p
      · simpa only [if_neg before, if_pos after, lookupEdge] using ihRight
      · simp only [if_neg before, if_neg after]

theorem edge_eq_lookup (graph : Graph) : graph.edge = graph.lookupEdge := by
  funext p q
  exact edge_eq_lookup_at graph p q

end ProjectEulerSolutions.PrimePairCertificate.Graph
