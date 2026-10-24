import ProjectEulerSolutions.Lib.PrimePairLarge.Graph024
namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate
def graph : Graph := node5903
theorem graph_covers : Budget.Covers PrimePairVertices.largeTree 26034 graph := node5903_covers
theorem graph_eq_data : graph = Data.graph := node5903_eq_data
theorem graph_coverage (p : Nat) (member : p ∈ PrimePairVertices.large) (competitive : 2 * p < 26034) : p ∈ graph.vertices := by
  rw [graph_eq_data]
  exact Data.coverage p member competitive
end ProjectEulerSolutions.PrimePairLarge
