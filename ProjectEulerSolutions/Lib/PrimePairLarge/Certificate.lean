import ProjectEulerSolutions.Lib.PrimePairLarge.Graph
import ProjectEulerSolutions.Lib.PrimePairLarge.Search086
namespace ProjectEulerSolutions.PrimePairLarge
theorem large_search_checked : CliqueSearch.search graph.lookupEdge 5 [] PrimePairVertices.large 0
    ⟨26034, []⟩ = ⟨26033, [13, 5197, 5701, 6733, 8389]⟩ := by
  rw [graph_eq_data]
  exact Search.largeSearch
end ProjectEulerSolutions.PrimePairLarge
