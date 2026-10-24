import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch263
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch264

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan3 : scan edge (search edge 4) 5 [] 0 tail3 best = best := by
  have cons : tail3 = 13 :: tail4 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates3_checked]
  change scan edge (search edge 4) 5 [] 0 tail4
    (if 4 ≤ rootCandidates3.length then search edge 4 [13] rootCandidates3 13 best else best) = best
  rw [if_pos (by decide), rootBranch3]
  exact outerScan4

end ProjectEulerSolutions.PrimePairUpper.Search
