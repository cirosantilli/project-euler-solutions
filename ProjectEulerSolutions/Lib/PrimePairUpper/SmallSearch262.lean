import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch261
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch262

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan5 : scan edge (search edge 4) 5 [] 0 tail5 best = best := by
  have cons : tail5 = 19 :: tail6 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates5_checked]
  change scan edge (search edge 4) 5 [] 0 tail6
    (if 4 ≤ rootCandidates5.length then search edge 4 [19] rootCandidates5 19 best else best) = best
  rw [if_pos (by decide), rootBranch5]
  exact outerScan6

end ProjectEulerSolutions.PrimePairUpper.Search
