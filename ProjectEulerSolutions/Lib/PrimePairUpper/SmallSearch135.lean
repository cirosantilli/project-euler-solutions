import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch134
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch135

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan269 : scan edge (search edge 4) 5 [] 0 tail269 best = best := by
  have cons : tail269 = 1747 :: tail270 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates269_checked]
  change scan edge (search edge 4) 5 [] 0 tail270
    (if 4 ≤ rootCandidates269.length then search edge 4 [1747] rootCandidates269 1747 best else best) = best
  rw [if_pos (by decide), rootBranch269]
  exact outerScan270

set_option maxRecDepth 4096 in
theorem outerScan268 : scan edge (search edge 4) 5 [] 0 tail268 best = best := by
  have cons : tail268 = 1741 :: tail269 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates268_checked]
  change scan edge (search edge 4) 5 [] 0 tail269
    (if 4 ≤ rootCandidates268.length then search edge 4 [1741] rootCandidates268 1741 best else best) = best
  rw [if_pos (by decide), rootBranch268]
  exact outerScan269

end ProjectEulerSolutions.PrimePairUpper.Search
