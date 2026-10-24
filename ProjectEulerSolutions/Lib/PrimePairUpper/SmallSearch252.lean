import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch251
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch252

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan15 : scan edge (search edge 4) 5 [] 0 tail15 best = best := by
  have cons : tail15 = 61 :: tail16 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates15_checked]
  change scan edge (search edge 4) 5 [] 0 tail16
    (if 4 ≤ rootCandidates15.length then search edge 4 [61] rootCandidates15 61 best else best) = best
  rw [if_pos (by decide), rootBranch15]
  exact outerScan16

end ProjectEulerSolutions.PrimePairUpper.Search
