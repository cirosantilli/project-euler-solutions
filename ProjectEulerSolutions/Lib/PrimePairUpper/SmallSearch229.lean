import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch228
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch229

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan48 : scan edge (search edge 4) 5 [] 0 tail48 best = best := by
  have cons : tail48 = 233 :: tail49 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates48_checked]
  change scan edge (search edge 4) 5 [] 0 tail49
    (if 4 ≤ rootCandidates48.length then search edge 4 [233] rootCandidates48 233 best else best) = best
  rw [if_pos (by decide), rootBranch48]
  exact outerScan49

end ProjectEulerSolutions.PrimePairUpper.Search
