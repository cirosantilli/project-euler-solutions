import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch266
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch267

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan0 : scan edge (search edge 4) 5 [] 0 tail0 best = best := by
  have cons : tail0 = 3 :: tail1 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates0_checked]
  change scan edge (search edge 4) 5 [] 0 tail1
    (if 4 ≤ rootCandidates0.length then search edge 4 [3] rootCandidates0 3 best else best) = best
  rw [if_pos (by decide), rootBranch0]
  exact outerScan1

theorem smallSearch : search edge 5 [] PrimePairVertices.small 0 best = best := by
  rw [search_succ_eq_scan, ← tail0_eq_vertices]
  exact outerScan0
theorem smallSearch_edge : search SearchData.graph.edge 5 [] PrimePairVertices.small 0 best = best := by
  rw [PrimePairCertificate.Graph.edge_eq_lookup]
  exact smallSearch

end ProjectEulerSolutions.PrimePairUpper.Search
