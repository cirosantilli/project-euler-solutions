import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch249
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch250

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan17 : scan edge (search edge 4) 5 [] 0 tail17 best = best := by
  have cons : tail17 = 71 :: tail18 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates17_checked]
  change scan edge (search edge 4) 5 [] 0 tail18
    (if 4 ≤ rootCandidates17.length then search edge 4 [71] rootCandidates17 71 best else best) = best
  rw [if_pos (by decide), rootBranch17]
  exact outerScan18

end ProjectEulerSolutions.PrimePairUpper.Search
