import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch260
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch261

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan6 : scan edge (search edge 4) 5 [] 0 tail6 best = best := by
  have cons : tail6 = 23 :: tail7 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates6_checked]
  change scan edge (search edge 4) 5 [] 0 tail7
    (if 4 ≤ rootCandidates6.length then search edge 4 [23] rootCandidates6 23 best else best) = best
  rw [if_pos (by decide), rootBranch6]
  exact outerScan7

end ProjectEulerSolutions.PrimePairUpper.Search
