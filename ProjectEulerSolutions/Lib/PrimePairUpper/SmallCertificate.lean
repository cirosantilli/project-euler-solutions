import ProjectEulerSolutions.Lib.PrimePairUpper.SmallGraph
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch267

namespace ProjectEulerSolutions.PrimePairUpper
theorem small_search_checked : CliqueSearch.search smallGraph.edge 5 [] PrimePairVertices.small 0
    ⟨10 ^ 30, []⟩ = ⟨10 ^ 30, []⟩ := by
  rw [smallGraph_eq_searchData]
  exact Search.smallSearch_edge
end ProjectEulerSolutions.PrimePairUpper
