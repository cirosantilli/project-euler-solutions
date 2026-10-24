import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch227
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch228

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan49 : scan edge (search edge 4) 5 [] 0 tail49 best = best := by
  have cons : tail49 = 239 :: tail50 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates49_checked]
  change scan edge (search edge 4) 5 [] 0 tail50
    (if 4 ≤ rootCandidates49.length then search edge 4 [239] rootCandidates49 239 best else best) = best
  rw [if_pos (by decide), rootBranch49]
  exact outerScan50

end ProjectEulerSolutions.PrimePairUpper.Search
