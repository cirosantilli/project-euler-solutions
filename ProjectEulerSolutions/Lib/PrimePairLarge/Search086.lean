import ProjectEulerSolutions.Lib.PrimePairLarge.Search085
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot000
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot001
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan1 : scan edge (search edge 4) 5 [] 0 tail1 seed = winner := by
  rw [tail1, scan, if_neg (by decide), root1Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail2
    (if 4 ≤ root1Candidates.length then search edge 4 [7] root1Candidates 7 seed else seed) = winner
  rw [if_pos (by decide), root1Branch]
  exact outerScan2
set_option maxRecDepth 4096 in
theorem outerScan0 : scan edge (search edge 4) 5 [] 0 tail0 seed = winner := by
  rw [tail0, scan, if_neg (by decide), root0Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail1
    (if 4 ≤ root0Candidates.length then search edge 4 [3] root0Candidates 3 seed else seed) = winner
  rw [if_pos (by decide), root0Branch]
  exact outerScan1
theorem largeSearch : search Data.graph.lookupEdge 5 [] PrimePairVertices.large 0 seed = winner := by
  rw [search_succ_eq_scan, ← tail0_eq_vertices]
  exact outerScan0
end ProjectEulerSolutions.PrimePairLarge.Search
